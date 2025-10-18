import numpy as np
import time

from scipy.optimize import minimize, differential_evolution 
from scipy.interpolate import interp1d

from Container import NNPredictionHist


class MPC:
    def __init__(self, model, predictionHorizon, controlSignalSampleTime=None):
        self.model = model
        self.predictionHorizon = predictionHorizon
        self.controlSignalSampleTime = model.sampleTimeNNInput if controlSignalSampleTime is None else controlSignalSampleTime

    def getNumberOfControlSignalsWithinPredictionHorizon(self):
        return int(np.floor(self.predictionHorizon / self.controlSignalSampleTime))

    def _mergeData(self, time1, data1, time2, data2):
        combined = dict(zip(time1, data1))
        if time2 is not None: combined.update(zip(time2, data2))

        timeCombined = sorted(combined)
        dataCombined = [combined[t] for t in timeCombined]

        return timeCombined, dataCombined

    def _getIncludedCoutFromMeasurementsAndPredictions(self, NN_time_vec, timeVec, includedCout, nnPredictionHist):
        est_Cout_times, est_Cout_mean_values,_ = nnPredictionHist.getMeanAndStd()
        timeCoutCombined, CoutCombined = self._mergeData(timeVec, includedCout, est_Cout_times, est_Cout_mean_values)
        CoutInterp = interp1d(timeCoutCombined, CoutCombined, kind='linear', fill_value="extrapolate")
        return CoutInterp(NN_time_vec)

    def _predictOutputs(self, currentTime, timeVec, Cin, flowRateTotal, temperature, includedCout, returnEntireContainer=False):
        nnPredictionHist = NNPredictionHist()
        deltaTimeNN = self.model.sampleTimeNNInput

        CInInterp = interp1d(timeVec, Cin, kind='linear', fill_value="extrapolate")
        TemperatureInterp = interp1d(timeVec, temperature, kind='linear', fill_value="extrapolate")
        flowRateTotalInterp = interp1d(timeVec, flowRateTotal, kind='linear', fill_value="extrapolate")

        for NNPredTimeStart in np.arange(currentTime+deltaTimeNN, currentTime + self.predictionHorizon+deltaTimeNN - self.model.timePrediction + 1e-3, deltaTimeNN):
            #assert NNPredTimeStart - self.model.timeBack >= 0, "NN prediction time start must be greater than time back"
            #print(f"Predicting from t = {NNPredTimeStart:.2f} to {(NNPredTimeStart+self.model.timePrediction):.2f} s")
            
            NN_time_vec = np.arange(NNPredTimeStart - self.model.timeBack, NNPredTimeStart, deltaTimeNN)
            NN_pred_time_vec = np.arange(NNPredTimeStart, NNPredTimeStart + self.model.timePrediction, deltaTimeNN)

            cin = CInInterp(NN_time_vec)[0:2, :]
            flr = flowRateTotalInterp(NN_time_vec)
            temp = TemperatureInterp(NN_time_vec)
            includedCoutInp = self._getIncludedCoutFromMeasurementsAndPredictions(NN_time_vec, timeVec, includedCout, nnPredictionHist)

            c_combi = np.array(cin[0, :]) * np.array(cin[1, :])
        
            X = np.vstack((cin, c_combi, flr, temp, includedCoutInp)).flatten()

            #print(f"\t{X}")
            Cpred = self.model.predictData(X)
            nnPredictionHist.append(NN_pred_time_vec, Cpred)

            #print(f"cin: {Cin[1, -1]}\nnnpred: {Cpred}\n")
        

        if returnEntireContainer: return nnPredictionHist

        nnPred_time, nnPred_CPredMean, nnPred_CPredStd = nnPredictionHist.getMeanAndStd()
        return nnPred_time, nnPred_CPredMean, nnPred_CPredStd
    
    def _costFunction(self, u, currentTime, prevInputs, prevOutputs, refSignals):
        # Control signals into the future
        # Cin1, Cin2, totalFlowRate, temperature
        u = u.reshape(4, -1)

        Next_Cin = u[0:2, :]
        Next_flowRateTotal = u[2, :]  
        Next_temperature = u[3, :]
        Next_time = np.linspace(currentTime, currentTime+self.predictionHorizon, self.getNumberOfControlSignalsWithinPredictionHorizon()+1)[1:]

        prevTimeIdx = np.where(prevInputs.time <= currentTime)[0]
        cin = prevInputs.Cin[0:2, prevTimeIdx]
        flowRate = prevInputs.flowRate[prevTimeIdx]
        temperature = prevInputs.temperature[prevTimeIdx]
        time = prevInputs.time[prevTimeIdx]

        _, Cin1 = self._mergeData(time, cin[0, :], Next_time, Next_Cin[0, :])
        _, Cin2 = self._mergeData(time, cin[1, :], Next_time, Next_Cin[1, :])
        Cin = np.vstack((Cin1, Cin2))

        _, flowRateTotal = self._mergeData(time, flowRate, Next_time, Next_flowRateTotal)
        timeVec, temperature = self._mergeData(time, temperature, Next_time, Next_temperature)

        includedCout = np.NaN * np.ones_like(timeVec) 
        for idx, species in enumerate(prevOutputs.Cout[2, prevTimeIdx]): includedCout[idx] = species # Filter out None values

        
        nnPred_time, nnPred_CPredMean, nnPred_CPredStd = self._predictOutputs(currentTime, timeVec, Cin, flowRateTotal, temperature, includedCout)

        CoutInterp = interp1d(nnPred_time, nnPred_CPredMean, kind='linear', fill_value="extrapolate")
        CoutPred = CoutInterp(Next_time)

        Cref = refSignals.getReferences(Next_time)

        uPlusPrev = np.hstack((np.array([np.hstack((prevInputs.Cin[0:2, -1], prevInputs.flowRate[-1], prevInputs.temperature[0]))]).T, u))
        smoothness_penalty = np.mean(np.diff(uPlusPrev, axis=1)**2)

        #print(f"CoutPred: {CoutPred}, u:")
        return 1e3*np.mean((Cref[2, :] - CoutPred)**2) + 0 * smoothness_penalty
        

    def _optimizeControlSignals(self, currentTime, prevInputs, prevOutputs, refSignals, printTime=False, initialCtrlSignalGuess=None):

        # Constraints and bounds can be defined here
        constraints = ()
        bounds = [(0.2, 1), (0.2, 1), (.3, 2), (150, 150.1)]  # C1, C2, flowRate, temperature
        extendedBounds = [bound for bound in bounds for _ in range(self.getNumberOfControlSignalsWithinPredictionHorizon())]

        # Initial guess for control signals
        if initialCtrlSignalGuess is None:
            initialCtrlSignalGuess = np.array([np.random.uniform(lower+.01, upper-.01) for lower, upper in bounds])
            initialCtrlSignalGuess = np.kron(initialCtrlSignalGuess, np.ones((1, self.getNumberOfControlSignalsWithinPredictionHorizon())))[0]

        startTime = time.time()
        result = minimize(
            self._costFunction,
            initialCtrlSignalGuess.ravel(),
            args=(currentTime, prevInputs, prevOutputs, refSignals),
            method='SLSQP',
            bounds=extendedBounds,
            constraints=constraints,
            #method='trust-constr',
            options={
                #'ftol': 1e-1,
                #'xtol': 1e-6,
                #'disp': True,
                #'maxiter': 300,
                #'maxfunccalls': 1000,
            }
        )
        
        #result = differential_evolution(
        #    self._costFunction,
        #    extendedBounds,
        #    args=(currentTime, prevInputs, prevOutputs, refSignals),
        #    strategy='best1bin', # 'best1bin', 'currenttobest1bin'
        #    mutation=(0.7, 1.5),  # (0.9, 1.9) (0.7, 1.5) Controls step sizes
        #    recombination=0.7,
        #    tol=0.05,
        #    maxiter=30,
        #    popsize=5,
        #    #workers=-1,
        #    disp=True
        #)


        endTime = time.time()
        if printTime: print(f"\t\tOptimization took {endTime - startTime:.2f} seconds ({((endTime - startTime)/60):.2f} minutes)")
        
        Next_time = np.linspace(currentTime, currentTime+self.predictionHorizon, self.getNumberOfControlSignalsWithinPredictionHorizon()+1)[1:]
        
        X = result.x.reshape(4, -1)
        Next_Cin = np.zeros((4, X.shape[1]))
        Next_Cin[0:2, :] = X[0:2, :]
        Next_flowRate = X[2, :]     
        Next_temperature = X[3, :]


        return Next_time, Next_Cin, Next_flowRate, Next_temperature 

    def getNextControlSignals(self, currentTime, prevInputs, prevOutputs, refSignals, printTime=False):
        return self._optimizeControlSignals(currentTime, prevInputs, prevOutputs, refSignals, printTime=printTime)
        