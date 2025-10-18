import numpy as np
import time as timeModule
from Container import OutputHist, InputHist, NNPredictionHist
from scipy.interpolate import interp1d

def getNumberOfConstRefsInCref(Cref):
    changesInCref = np.abs(np.hstack((0, np.diff(Cref)))) > 0.05
    numberOfStepsInCref = np.sum(np.diff(changesInCref) > 0) // 2
    numberOfConstIntervalsInCref = numberOfStepsInCref + 1
    return numberOfConstIntervalsInCref

def getChangesInCref(Cref, flowRate, deltaT, reactorVolumeIn_mL):
    changesInCref = np.hstack((0, np.diff(1 * (np.abs(np.hstack((0, np.diff(Cref)))) > 0.05))))
    changesInCref[changesInCref < 0] = 0

    indices = [i for i, val in enumerate(changesInCref) if val == 1]
    
    
    flowRatesAtSteps = flowRate[changesInCref == 1] # mL/min
    flowRatesAtSteps[np.abs(flowRatesAtSteps) < 1e-4] = 1
    shifts = np.floor(reactorVolumeIn_mL / flowRatesAtSteps * 60 / deltaT)

    shiftedChangesInCred = np.zeros_like(Cref)

    for idx, shift in zip(indices, shifts):
        new_idx = max(0, idx - shift)  # Absichern, dass Index >= 0 bleibt
        shiftedChangesInCred[int(new_idx)] = 1

    return shiftedChangesInCred, changesInCref


def designSpaceExploration_fixedTemperature(myNN, outputHist, timeVec, Cref, overlapInPercent, temperature, Cin1Bounds, Cin2Bounds, flowRateBounds, samples, numberOfFlowRateSamples=None, verbose=False):

    numberOfCinSamples = samples
    if numberOfFlowRateSamples is None: numberOfFlowRateSamples = samples
    numberOfTempSamples = 1   

    iteration, totalIterations = 1, numberOfCinSamples**2 * numberOfTempSamples * numberOfFlowRateSamples

    C1_vals, C2_vals = np.array([]), np.array([])
    flowRate_vals = np.array([])
    costs = np.array([])
    execTime_vals = np.array([])
    
    tempVec = temperature*np.ones_like(timeVec)

    for Cin1 in np.linspace(Cin1Bounds[0], Cin1Bounds[1], numberOfCinSamples):
        for Cin2 in np.linspace(Cin2Bounds[0], Cin2Bounds[1], numberOfCinSamples):
            for flr in np.linspace(flowRateBounds[0], flowRateBounds[1], numberOfFlowRateSamples):

                Cin = np.zeros((4, len(timeVec)))
                Cin[0, :] = Cin1
                Cin[1, :] = Cin2
                
                flowRate = flr*np.ones_like(timeVec)
                cost, execTime = costFunction(myNN, outputHist, timeVec, Cin, flowRate, tempVec, Cref, overlapInPercent)
                execTime_vals = np.hstack((execTime_vals, execTime))

                C1_vals = np.hstack((C1_vals, Cin1))
                C2_vals = np.hstack((C2_vals, Cin2))
                flowRate_vals = np.hstack((flowRate_vals, flr))
                costs = np.hstack((costs, cost))

                expTime = np.mean(execTime_vals) * (totalIterations - iteration + 1)
                if verbose: print(f"[{iteration}/{totalIterations} Exp.Time: {expTime:.2f}s ({(expTime/60):.0f}min)]\tC1: {Cin1:.1f}\tC2: {Cin2:.1f}\Flow rate: {flr:.2f}mL/min\tTime: {execTime:.2f}s\tCost: {cost:.3f}")
                iteration+=1

    return {
        "C1_vals": C1_vals,
        "C2_vals": C2_vals,
        "temp": temperature,
        "flowRate_vals": flowRate_vals,
        "execTime_vals": execTime_vals,
        "costs": costs,
        "numberOfCinSamples":numberOfCinSamples,
        "numberOfFlowRateSamples": numberOfFlowRateSamples,
        "numberOfTempSamples": numberOfTempSamples
    }


def designSpaceExploration_fixedFlowRate(myNN, outputHist, timeVec, Cref, overlapInPercent, temperatureBounds, Cin1Bounds, Cin2Bounds, flowRate, samples, numberOfTempSamples=None, verbose=False):

    numberOfCinSamples = samples
    if numberOfTempSamples is None: numberOfTempSamples = samples
    numberOfFlowRateSamples = 1   

    iteration, totalIterations = 1, numberOfCinSamples**2 * numberOfTempSamples * numberOfFlowRateSamples

    C1_vals, C2_vals = np.array([]), np.array([])
    temperature_vals = np.array([])
    costs = np.array([])
    execTime_vals = np.array([])
    
    flowRateVec = flowRate*np.ones_like(timeVec)

    for Cin1 in np.linspace(Cin1Bounds[0], Cin1Bounds[1], numberOfCinSamples):
        for Cin2 in np.linspace(Cin2Bounds[0], Cin2Bounds[1], numberOfCinSamples):
            for temp in np.linspace(temperatureBounds[0], temperatureBounds[1], numberOfTempSamples):

                Cin = np.zeros((4, len(timeVec)))
                Cin[0, :] = Cin1
                Cin[1, :] = Cin2
                
                tempVec = temp*np.ones_like(timeVec)

                cost, execTime = costFunction(myNN, outputHist, timeVec, Cin, flowRateVec, tempVec, Cref, overlapInPercent)
                execTime_vals = np.hstack((execTime_vals, execTime))

                C1_vals = np.hstack((C1_vals, Cin1))
                C2_vals = np.hstack((C2_vals, Cin2))
                temperature_vals = np.hstack((temperature_vals, temp))
                costs = np.hstack((costs, cost))

                expTime = np.mean(execTime_vals) * (totalIterations - iteration + 1)
                if verbose: print(f"[{iteration}/{totalIterations} Exp.Time: {expTime:.2f}s ({(expTime/60):.0f}min)]\tC1: {Cin1:.1f}\tC2: {Cin2:.1f}\Flow rate: {flr:.2f}mL/min\tTime: {execTime:.2f}s\tCost: {cost:.3f}")
                iteration+=1

    return {
        "C1_vals": C1_vals,
        "C2_vals": C2_vals,
        "temp_vals": temperature_vals,
        "flowRate": flowRate,
        "execTime_vals": execTime_vals,
        "costs": costs,
        "numberOfCinSamples":numberOfCinSamples,
        "numberOfFlowRateSamples": numberOfFlowRateSamples,
        "numberOfTempSamples": numberOfTempSamples
    }

def designSpaceExploration_fixedTemperatureAndFixedFlowRate(myNN, outputHist, timeVec, Cref, overlapInPercent, temperature, flowRate, Cin1Bounds, Cin2Bounds, samples, numberOfFlowRateSamples=None, verbose=False, prevXopt=None):

    numberOfCinSamples = samples
    numberOfFlowRateSamples = 1
    numberOfTempSamples = 1   

    iteration, totalIterations = 1, numberOfCinSamples**2 * numberOfTempSamples * numberOfFlowRateSamples

    C1_vals, C2_vals = np.array([]), np.array([])
    costs = np.array([])
    execTime_vals = np.array([])
    
    tempVec = temperature*np.ones_like(timeVec)

    for Cin1 in np.linspace(Cin1Bounds[0], Cin1Bounds[1], numberOfCinSamples):
        for Cin2 in np.linspace(Cin2Bounds[0], Cin2Bounds[1], numberOfCinSamples):

            Cin = np.zeros((4, len(timeVec)))
            Cin[0, :] = Cin1
            Cin[1, :] = Cin2
            
            flowRateVec = flowRate*np.ones_like(timeVec)
            cost, execTime = costFunction(myNN, outputHist, timeVec, Cin, flowRateVec, tempVec, Cref, overlapInPercent, prevXopt=prevXopt)
            execTime_vals = np.hstack((execTime_vals, execTime))

            C1_vals = np.hstack((C1_vals, Cin1))
            C2_vals = np.hstack((C2_vals, Cin2))
            costs = np.hstack((costs, cost))

            expTime = np.mean(execTime_vals) * (totalIterations - iteration + 1)
            if verbose: print(f"[{iteration}/{totalIterations} Exp.Time: {expTime:.2f}s ({(expTime/60):.0f}min)]\tC1: {Cin1:.1f}\tC2: {Cin2:.1f}\Flow rate: {flr:.2f}mL/min\tTime: {execTime:.2f}s\tCost: {cost:.3f}")
            iteration+=1

    return {
        "C1_vals": C1_vals,
        "C2_vals": C2_vals,
        "temp": temperature,
        "flowRate": flowRate,
        "execTime_vals": execTime_vals,
        "costs": costs,
        "numberOfCinSamples":numberOfCinSamples,
        "numberOfFlowRateSamples": numberOfFlowRateSamples,
        "numberOfTempSamples": numberOfTempSamples
    }

def findMinimumCost(dSEResult):
    # Simple opt
    min_idx = np.argmin(dSEResult.get("costs"))
    opt_C1 = dSEResult.get("C1_vals")[min_idx]
    opt_C2 = dSEResult.get("C2_vals")[min_idx]
    opt_cost = dSEResult.get("costs")[min_idx]

    if "flowRate_vals" in dSEResult: 
        opt_flowRate = dSEResult.get("flowRate_vals")[min_idx]
    elif "flowRate" in dSEResult:
        opt_flowRate = dSEResult.get("flowRate")
    else:
        raise("Oha")
    
    if "temp_vals" in dSEResult: 
        opt_T = dSEResult.get("temp_vals")[min_idx]
    elif "temp" in dSEResult:
        opt_T = dSEResult.get("temp")
    else:
        raise("Oha")
    
    #print(f"Optimales Setup:\n  C1 = {opt_C1:.3f}\n  C2 = {opt_C2:.3f}\n  T = {opt_T:.1f} °C\n  fR: = {opt_flowRate} mL/min\n  Cost = {opt_cost:.4f}")
 
    return {
        "C1": opt_C1,
        "C2": opt_C2,
        "temp": opt_T,
        "flowRate": opt_flowRate,
        "cost": opt_cost
    }

def getRegionAroundOptimum(result, rangeSizeInPercentage=50):
    costs = result.get("costs")

    costRange = max(costs) - min(costs)
    min_cost = min(costs)
    threshold = min_cost + rangeSizeInPercentage * costRange / 100
    indicesPercentCostRange = [idx for idx, cost in enumerate(costs) if cost <= threshold]

    minmax = lambda data_: (min(data_), max(data_))

    if "flowRate_vals" in result: 
        newFlowRateRange = minmax(result.get("flowRate_vals")[indicesPercentCostRange])
    elif "flowRate" in result:
        newFlowRateRange = result.get("flowRate")
    else:
        raise("Oha")
    
    if "temp_vals" in result: 
        newTemperatureRange = minmax(result.get("temp_vals")[indicesPercentCostRange])
    elif "temp" in result:
        newTemperatureRange = result.get("temp")
    else:
        raise("Oha")

    newC1Range = minmax(result.get("C1_vals")[indicesPercentCostRange])
    newC2Range = minmax(result.get("C2_vals")[indicesPercentCostRange])

    return {
        "newFlowRateRange": newFlowRateRange,
        "newTemperatureRange": newTemperatureRange,
        "newC1Range": newC1Range,
        "newC2Range": newC2Range
    }


def getTimeVectorsForNN(myNN, currentTime):
    NN_time_vec = np.arange(currentTime - myNN.timeBack, currentTime, myNN.sampleTimeNNInput) + myNN.sampleTimeNNInput
    NN_pred_time_vec = np.arange(currentTime, currentTime + myNN.timePrediction, myNN.sampleTimeNNInput) + myNN.sampleTimeNNInput
    return NN_time_vec, NN_pred_time_vec

def getInputDataForNN(NN_time_vec, timeVec, Cin, flowRateTotal, temperature, timeVecIncludedCout, includedCout):
    Cin = interp1d(timeVec, Cin, kind='linear', fill_value="extrapolate")(NN_time_vec)
    flowRateTotal = interp1d(timeVec, flowRateTotal, kind='linear', fill_value="extrapolate")(NN_time_vec)
    temperature = interp1d(timeVec, temperature, kind='linear', fill_value="extrapolate")(NN_time_vec)
    includedCout = interp1d(timeVecIncludedCout, includedCout, kind='linear', fill_value="extrapolate")(NN_time_vec)

    c_combi = np.array(Cin[0, :]) * np.array(Cin[1, :])

    X = np.vstack((Cin, c_combi, flowRateTotal, temperature, includedCout)).flatten()
    return X


def simulateReactorUsingNN(myNN, outputHist, timeVec, Cin, flowRate, temperature, overlapInPercent=None):
    #Cin = np.zeros((4, len(timeVec)))
    timeVecIncludedCout = np.hstack((np.array([-1*myNN.timeBack, -.1]), outputHist.time))
    includedCout = np.hstack((np.array([0, 0]), outputHist.Cout[2, :] if outputHist.Cout.shape[0] > 0 else np.array([])))

    predHist = NNPredictionHist()


    if overlapInPercent is None:
        skipIdx = 1
    else:
        predictionHorizonNumberOfIdx = np.sum(1 * ((timeVec - timeVec[0]) < myNN.timePrediction))
        skipIdx = int((100 - overlapInPercent) * predictionHorizonNumberOfIdx / 100) 

    skipIdx = min(max(1, skipIdx), predictionHorizonNumberOfIdx) 

    for currentTime in timeVec[::skipIdx]:
        NN_time_vec, NN_pred_time_vec = getTimeVectorsForNN(myNN, currentTime)
        
        X = getInputDataForNN(NN_time_vec, timeVec, Cin[0:2, :], flowRate, temperature, timeVecIncludedCout, includedCout)
        Cpred = myNN.predictData(X)
        predHist.append(NN_pred_time_vec, Cpred)
        
        #timeVecIncludedCout = np.hstack((timeVecIncludedCout, currentTime + myNN.sampleTimeNNInpu))
        #includedCout = np.hstack((includedCout, Cpred[0]))

        CpredMeanFromHist = predHist.getMeanForTime(currentTime + myNN.sampleTimeNNInput)
        timeVecIncludedCout = np.hstack((timeVecIncludedCout, currentTime + myNN.sampleTimeNNInput))
        includedCout = np.hstack((includedCout, CpredMeanFromHist))

    return predHist

def costFunction(myNN, outputHist, timeVec, Cin, flowRate, temperature, Cref, overlapInPercent=False, plotResult=False, prevXopt=None):

    # Simulate NN for a given input
    startTime = timeModule.time()
    predHist = simulateReactorUsingNN(myNN, outputHist, timeVec, Cin, flowRate, temperature, overlapInPercent)
    execTime = timeModule.time() - startTime

    unique_times, mean_values, std_devs = predHist.getMeanAndStd()

    # Calculate error over prediction vs. reference
    interpPredictionCout = interp1d(unique_times, mean_values)(timeVec[1:]) 
    # We can do it over the entire prediction horizon yet as we only have one input concentration for the entire range
    # we can not cover/represent changes but would optimize for the mean value
    # Thus we take the first value only
    
    #deltaRef = interpPredictionCout - Cref[1:]
    deltaRef = interpPredictionCout - Cref[1]

    error = deltaRef**2

    if plotResult:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.plot(timeVec/60, Cref, 'r--', label=f'Cref')
        
        predTime, predMeanValues, predStdDevs = predHist.getMeanAndStd()
        ax.plot(predTime/60, predMeanValues, color='orange', label=f'Predicted Cout by NN - Mean')
        ax.fill_between(predTime/60, predMeanValues - predStdDevs, predMeanValues + predStdDevs, color='orange', alpha=0.2, label=f'Predicted Cout by NN - Std')
        
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Concentration')
        ax.legend()

        ax = fig.add_subplot(212)
        ax.plot(timeVec[1:]/60, error, 'g--', label=f'Cin2')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Error')
        ax.legend()

    totalError = np.sum(error)
    
    consumption = np.sum((Cin[:,1:] * np.diff(timeVec)), axis=1) / (timeVec[-1] - timeVec[0])
    economicCost = consumption[0] + 10 * consumption[1] + (np.average(temperature) / 3000)

    alteredCost = 0
    if prevXopt is not None:
        costRelToPrevInput = lambda data_, label_: np.sqrt(np.sum((data_ - prevXopt.get(label_))**2)) / len(Cin[0, :])
        alteredCost = costRelToPrevInput(Cin[0, :], "C1") + costRelToPrevInput(Cin[1, :], "C2") + costRelToPrevInput(flowRate, "flowRate") + costRelToPrevInput(temperature, "temp")

    cost = 10 * totalError + economicCost + 50 * alteredCost
    

    return cost, execTime