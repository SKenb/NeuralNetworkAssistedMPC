# main_testNNOptimizationWithFeedback_WithDoE_WithMeasurementUncertainties.py

import matplotlib.pyplot as plt
import numpy as np
import pickle

from enum import Enum

from NN import NNWithContext
from Container import OutputHist, InputHist, NNPredictionHist
from scipy.interpolate import interp1d

from DesignSpaceExploration import getRegionAroundOptimum, designSpaceExploration_fixedTemperature, designSpaceExploration_fixedTemperatureAndFixedFlowRate, designSpaceExploration_fixedFlowRate, findMinimumCost, getChangesInCref
from CommonPlots import plotDesignSpaceExploration, plotResultsOverDesignSpaceExplorationIterationsVeryCOnly
from Common import MODE, SENTINEL, getRealReactor, getReactorModel
from Polisher import POLISHER, polishOptimum, prepareIntegratingStateForPolisher, getIntegratedStateOffset
from Sentinel import polishOptWithMPCIntegratingSentinel

#mode = MODE.C1C2
#polisher = POLISHER.COBYLA
#sentinel = SENTINEL.MPC_INTEGRATING_PBM
#showPlots = False
#measurementUncertainties = True

mode = MODE.C1C2
polisher = POLISHER.COBYLA
sentinel = SENTINEL.NONE
showPlots = False
measurementUncertainties = False

def getCinWithSteps(timeVec):
    Cin = np.zeros((4, len(timeVec)))

    stepValues = [(1, 0), (1, .7), (1, .2), (1, 1)]

    for idx, stepValue in enumerate(stepValues):
        stepIdx = int(idx * Cin.shape[1] / len(stepValues))
        Cin[0, stepIdx:] = stepValue[0]
        Cin[1, stepIdx:] = stepValue[1]

    return Cin

sentinelPIDController = None
def getSentinelPIDController(sample_time):
    global sentinelPIDController
    if sentinelPIDController is not None: return sentinelPIDController

    sentinelPIDController = PIDController(
        #Kp=0.5, Ki=0.005, Kd=0,
        Kp=0.7, Ki=0.01, Kd=0,
        sample_time=sample_time,
        output_limits=(-.2, .2)
    )

    return sentinelPIDController



def main():

    reactorModel = getReactorModel(getModelForOptimization=False)
    realReactor = getRealReactor(useAlteredParameters=measurementUncertainties)
    myNN = NNWithContext.LOAD("./B1_Python/A1_NNAssistedMPC/Dumps/NNWithContextOptParams.nn")

    sampleTimeMPC = 10*60
    predictionHorizon = 30*60

    match sentinel:
        case SENTINEL.NONE: sampleTimeSentinel = sampleTimeMPC
        case SENTINEL.PI_BAND_C1: sampleTimeSentinel = 10
        case SENTINEL.MPC_INTEGRATING_PBM: sampleTimeSentinel = 2*60
        case SENTINEL.MPC_INTEGRATING_MAP_PBM: sampleTimeSentinel = 2*60
        case SENTINEL.MPC_INTEGRATING_MAP_WITH_INTERPOLATION_PBM: sampleTimeSentinel = 2*60
        case _: raise ValueError("Unknown sentinel")

    tEnd = 7*sampleTimeMPC + predictionHorizon
    deltaTsim = 2*myNN.sampleTimeNNInput
    timeVec = np.arange(0, tEnd, deltaTsim)

    C3ref = .6*np.ones_like(timeVec)
    C3ref[timeVec >= 2*sampleTimeMPC] = .3
    C3ref[timeVec >= 3*sampleTimeMPC] = .8
    C3ref[timeVec >= 5*sampleTimeMPC] = .4

    tempRef = 150*np.ones_like(timeVec)
    fixedFlowRate = 1

    ## Plot References
    fig1 = plt.figure()
    ax = fig1.add_subplot(211)
    ax.plot(timeVec/60, C3ref, color='green', label=f'Reference Cout')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration')
    ax.legend()

    ax = fig1.add_subplot(212)
    ax.plot(timeVec/60, tempRef, color='green', label=f'Reference Temp')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature')
    ax.legend()

    ## Time loop / MPC

    outputHistReactorModel = OutputHist()
    outputHist = OutputHist()
    inputHist = InputHist()

    prevCref = -10

    mpcSampleTimes = np.arange(0, timeVec[-1]-predictionHorizon, sampleTimeMPC)
    sentinelSampleTimes = np.arange(0, timeVec[-1]-predictionHorizon, sampleTimeSentinel)

    ## Add time points where Cref changes
    shiftedChangesInCred, changesInCref = getChangesInCref(C3ref, fixedFlowRate*np.ones_like(timeVec), deltaTsim, realReactor.getVolumeIn_mL())
    timesOfCrefChange = timeVec[shiftedChangesInCred == 1]
    numberOfConstCrefs = int(np.sum(shiftedChangesInCred) + 1)

    mpcSampleTimes = np.hstack((mpcSampleTimes, timesOfCrefChange))
    mpcSampleTimes.sort()

    allSampleTimes = np.unique(np.hstack((mpcSampleTimes, sentinelSampleTimes)))
    allSampleTimes.sort()

    opt = None

    for sampleTimeIdx, currentTime in enumerate(allSampleTimes):
        isMPCOptimizationTimePoint = currentTime in mpcSampleTimes
        #if currentTime >= 700: break # Simulate only for a short time

        if not isMPCOptimizationTimePoint:
            print(f"Time: {currentTime/60} min ({currentTime}s)\tSentinel only")
        else:
            predTimeVec = np.arange(currentTime, currentTime+predictionHorizon+deltaTsim, deltaTsim)
            print(f"Time: {currentTime/60} min ({currentTime}s)\tPredict until: {predTimeVec[-1]/60}min")

            C3ref_predH = interp1d(timeVec, C3ref, kind='linear')(predTimeVec)
            tempRef_predH = interp1d(timeVec, tempRef, kind='linear')(predTimeVec)

            meanRT = realReactor.getVolumeIn_mL() / fixedFlowRate
            C3ref_predH_NoPreview = C3ref_predH[0]*np.ones_like(C3ref_predH)
            C3ref_predH_Preview = C3ref_predH[int(60 * meanRT / deltaTsim)]*np.ones_like(C3ref_predH)
            C3ref_toUse = C3ref_predH_Preview
            

            ## Do the optimization
            if np.abs(prevCref - C3ref_toUse[0]) > .1: # if True:
                print(f"\tStarting optimization - global search")
                # NEW Cref - do global search
                Cin1Bounds, Cin2Bounds = (0, 1), (0, 1)
                flowRateBounds = (0.2, 2)
                temperatureBounds = (50, 200)
                temperature = tempRef_predH[0]
                samples = 5
                overlap = 20
                rangeSizeInPercentage = 50
                maxIterations = 8
                includeCostUsingPrevXOpt = False

                getSentinelPIDController(sampleTimeSentinel).reset()

            else:
                print(f"\tStarting optimization - around previous optimum")
                aroundOldBound = lambda oldBounds_, border_, min_, max_: (max((oldBounds_[0] - border_), min_), min((oldBounds_[1] + border_), max_)) 
                # Cref quite similar - look in sourounding
                Cin1Bounds = aroundOldBound(Cin1Bounds, .2, 0, 1)
                Cin2Bounds = aroundOldBound(Cin2Bounds, .2, 0, 1)
                flowRateBounds = aroundOldBound(flowRateBounds, .4, 0.2, 2)
                temperatureBounds = aroundOldBound(temperatureBounds, 50, 50, 200)
                temperature = tempRef_predH[0]
                samples = 5
                overlap = 40
                rangeSizeInPercentage = 80
                maxIterations = 4
                includeCostUsingPrevXOpt = True

            prevCref = C3ref_predH_NoPreview[0]
            prevXopt = -100 * np.array([1, 1, 1, 1])

            results = []
            overlapHist = []
            rangeSizeInPercentageHist = []
            xOptMoveHist = []

            boundsDelta = lambda bounds_: bounds_[1] - bounds_[0]
            distanceBreakCritera = False

            while (boundsDelta(Cin1Bounds) > .1 or boundsDelta(Cin2Bounds) > .1) and maxIterations >= len(overlapHist) and not distanceBreakCritera:
                overlapHist.append(overlap)
                rangeSizeInPercentageHist.append(rangeSizeInPercentage)

                match mode:
                    case MODE.C1C2:
                        result = designSpaceExploration_fixedTemperatureAndFixedFlowRate(myNN, outputHist, predTimeVec, C3ref_toUse, overlap, temperature, fixedFlowRate, Cin1Bounds, Cin2Bounds, samples, verbose=False, prevXopt=opt if includeCostUsingPrevXOpt else None)
                    case MODE.C1C2FR:
                        result = designSpaceExploration_fixedTemperature(myNN, outputHist, predTimeVec, C3ref_toUse, overlap, temperature, Cin1Bounds, Cin2Bounds, flowRateBounds, samples, numberOfFlowRateSamples=None, verbose=False)
                    case MODE.C1C2TEMP:
                        result = designSpaceExploration_fixedFlowRate(myNN, outputHist, predTimeVec, C3ref_toUse, overlap, temperatureBounds, Cin1Bounds, Cin2Bounds, fixedFlowRate, samples, numberOfTempSamples=None, verbose=False) 
                    case _:
                        raise("Unknown mode") 
                
                
                results.append(result)
            
                print(f"\tdSE\tTemp: {temperatureBounds if mode == MODE.C1C2TEMP else temperature}\tFR: {flowRateBounds if mode == MODE.C1C2FR else fixedFlowRate}\tC1: {Cin1Bounds} ({boundsDelta(Cin1Bounds)})\tC2: {Cin2Bounds} ({boundsDelta(Cin2Bounds)})\tOverlap: {overlap}%\tRange: {rangeSizeInPercentage}%")
        
                newRanges = getRegionAroundOptimum(results[-1], rangeSizeInPercentage)

                Cin1Bounds, Cin2Bounds = newRanges.get("newC1Range"), newRanges.get("newC2Range")
                #flowRateBounds = newRanges.get("newFlowRateRange")
                overlap += 10 #20, 10 <-- faster
                overlap = min(overlap, 100)
                rangeSizeInPercentage *= .75 if rangeSizeInPercentage >= 10 else .5

                opt = findMinimumCost(result)
                Xopt = np.array([opt.get("C1"), opt.get("C2"), opt.get("temp"), opt.get("flowRate")])
                xOptMoveHist.append(np.sqrt(np.sum((prevXopt - Xopt)**2)))
                prevXopt = Xopt

                distanceBreakCritera = len(xOptMoveHist) > 4 and np.sum(xOptMoveHist[-1:-5:-1])/5 < 0.05

            
            if showPlots:
                plotResultsOverDesignSpaceExplorationIterationsVeryCOnly(results, mode)

            # Do detailed optimization around found dSE optimum using a 
            # inaccurate AD model
            print(f"\tStart polishing")

            C3RefValue = C3ref_toUse[0]

            opt = findMinimumCost(results[-1])

            MinMaxC1C2 = (0,1)
            MinMaxFlowRate = (0.2,2) if mode == MODE.C1C2FR else (opt.get("flowRate"), opt.get("flowRate"))
            MinMaxTemp = (50,200) if mode == MODE.C1C2TEMP else (opt.get("temp"), opt.get("temp"))

            polishedOpt = polishOptimum(mode, polisher, sentinel, opt, currentTime, inputHist, deltaTsim, C3RefValue, MinMaxC1C2, MinMaxFlowRate, MinMaxTemp, outputHist)

            C1, C2 = polishedOpt.get("C1"), polishedOpt.get("C2")
            flowRate = polishedOpt.get("flowRate") * np.ones_like(timeVec)
            temperature = polishedOpt.get("temp") * np.ones_like(timeVec)

            print(f"\tPolished optimum: C1: {C1:.3f} (Δ{polishedOpt.get('delta_C1'):.3f})\tC2: {C2:.3f} (Δ{polishedOpt.get('delta_C2'):.3f})\tT: {temperature[0]:.1f}°C\tfR: {flowRate[0]:.2f}mL/min\tCost: {polishedOpt.get('cost'):.4f}\tSuccess: {polishedOpt.get('success')} (nfev: {polishedOpt.get('nfev')})")

            
        # Sentinel
        deltaC1 = 0

        if sentinel is not SENTINEL.NONE:
            print("\tApply Sentinel")

            if len(outputHist.Cout) > 0:
                measuredC3 = outputHist.Cout[2, :];
                measuredC3 = measuredC3 if len(measuredC3) < 10 else measuredC3[-10:]
                measuredC3_Mean = np.mean(measuredC3)
                 
                match sentinel:
                    case SENTINEL.NONE:
                        raise ValueError("Sentinel is NONE - You must not be here @all")
                    case SENTINEL.PI_BAND_C1:
                        # Apply PI control
                        if np.all(np.abs(measuredC3 - C3ref_toUse[0]) < 0.1):
                            controller = getSentinelPIDController(sampleTimeSentinel)
                            deltaC1 = controller.getControlSignal(C3ref_toUse[0], measuredC3_Mean)
                            print(f"\tPID Band Sentinel - ΔC1: {deltaC1:.3f}")
                        else:
                            print(f"\tPID Band Sentinel - Band not reached")


                    case SENTINEL.MPC_INTEGRATING_PBM | SENTINEL.MPC_INTEGRATING_MAP_PBM | SENTINEL.MPC_INTEGRATING_MAP_WITH_INTERPOLATION_PBM:
                        
                        useIntegratorMap = sentinel == SENTINEL.MPC_INTEGRATING_MAP_PBM or sentinel == SENTINEL.MPC_INTEGRATING_MAP_WITH_INTERPOLATION_PBM
                        currentC3RefValue = interp1d(timeVec, C3ref, 'linear')(currentTime)

                        polishedOpt = polishOptWithMPCIntegratingSentinel(mode, polisher, sentinel, opt, currentTime, inputHist, deltaTsim, C3RefValue, MinMaxC1C2, MinMaxFlowRate, MinMaxTemp, outputHist, outputHistReactorModel, useIntegratorMap=useIntegratorMap, currentC3RefValue=currentC3RefValue)

                        C1, C2 = polishedOpt.get("C1"), polishedOpt.get("C2")
                        flowRate = polishedOpt.get("flowRate") * np.ones_like(timeVec)
                        temperature = polishedOpt.get("temp") * np.ones_like(timeVec)
                        print(f"\tPolished optimum (using measured data): C1: {C1:.3f} (Δ{polishedOpt.get('delta_C1'):.3f})\tC2: {C2:.3f} (Δ{polishedOpt.get('delta_C2'):.3f})\tT: {temperature[0]:.1f}°C\tfR: {flowRate[0]:.2f}mL/min\tCost: {polishedOpt.get('cost'):.4f}\tSuccess: {polishedOpt.get('success')} (nfev: {polishedOpt.get('nfev')})")
    

        # Prepare input signal
        Cin = np.array([(C1 + deltaC1)*np.ones_like(timeVec), C2*np.ones_like(timeVec), np.zeros_like(timeVec), np.zeros_like(timeVec)])
        
        # Prepare simulations
        simTimeStart = currentTime
        simTimeEnd = allSampleTimes[sampleTimeIdx+1] if sampleTimeIdx < len(allSampleTimes)-1 else currentTime+sampleTimeMPC

        # Simulate Reactor model
        print(f"\tSimulate reactor model for {Cin[0, 0]:.2f}, {Cin[1, 0]:.2f}, {temperature[0]:.1f}°C, {flowRate[0]:.2f}mL/min")

        timeReactorModel, CoutReactorModel, CspatialReactorModel = reactorModel.simulateStep(
            simTimeStart, simTimeEnd,
            timeVec, Cin, flowRate, temperature
        )
        outputHistReactorModel.append(timeReactorModel, CoutReactorModel, CspatialReactorModel)

        # Simulate Reactor
        print(f"\tSimulate real reactor for {Cin[0, 0]:.2f}, {Cin[1, 0]:.2f}, {temperature[0]:.1f}°C, {flowRate[0]:.2f}mL/min")

        time, Cout, Cspatial = realReactor.simulateStep(
            simTimeStart, simTimeEnd,
            timeVec, Cin, flowRate, temperature
        )

        myInputInterp = lambda data_: interp1d(timeVec, data_, kind='linear', fill_value="extrapolate")(time)

        outputHist.append(time, Cout, Cspatial)
        inputHist.append(time, myInputInterp(Cin), myInputInterp(flowRate), myInputInterp(temperature))

        print(f"\t(i) Integral State Offset: {getIntegratedStateOffset()}")

        if showPlots:
            realReactor.plot(additionalTracesToPlot=[
                    {"x": predTimeVec, "y": np.array([
                        np.NaN*np.zeros_like(predTimeVec),
                        np.NaN*np.zeros_like(predTimeVec),
                        C3ref_predH,
                        np.NaN*np.zeros_like(predTimeVec)
                    ]), "label": "Cref - Preview", "linestyle": "-"},
                    {"x": predTimeVec, "y": np.array([
                        np.NaN*np.zeros_like(predTimeVec),
                        np.NaN*np.zeros_like(predTimeVec),
                        C3ref_predH_NoPreview,
                        np.NaN*np.zeros_like(predTimeVec)
                    ]), "label": "Cref - NO Preview", "linestyle": "--"},
                ], showPlot=True)

    inputHist.plot()
    outputHist.plot()

    if sentinel == SENTINEL.PI_BAND_C1:
        getSentinelPIDController(sampleTimeSentinel).plotStateTrace(showPlot=True)

    ## Save data for comparison
    filename = f"dataDSE_{polisher.name if polisher is not SENTINEL.NONE else 'No'}-Polisher_{sentinel.name if sentinel is not SENTINEL.NONE else 'No'}-Sentinel_{'WithMeasurementUncertainties' if measurementUncertainties else 'NoMeasurementUncertainties'}.pkl"
    
    if True:
        with open(filename, "wb") as f:
            pickle.dump({
                "name": f"DSE - Polisher: {polisher.name}, Sentinel: {sentinel.name}, MeasurementUncertainties: {measurementUncertainties}",
                "description": f"DSE with NN only - Polisher: {polisher.name}, Sentinel: {sentinel.name}, MeasurementUncertainties: {measurementUncertainties}",
                "inputHist": inputHist,
                "outputHist": outputHist,
                "C3ref": (timeVec, C3ref),
                "mode": mode.name,
                "polisher": polisher.name,
                "sentinel": sentinel.name
            }, f)

    print("END")
    outputHist.createGif(timeVec, np.array([np.nan*np.ones_like(timeVec), np.nan*np.ones_like(timeVec), C3ref, np.nan*np.ones_like(timeVec)]))
    
    print("adf")




if __name__ == "__main__":
    main()