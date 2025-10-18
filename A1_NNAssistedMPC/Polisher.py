import numpy as np
from enum import Enum
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d 

from Common import MODE, SENTINEL, getReactorModel

class POLISHER(Enum):
    NONE = 0
    SLSQP = 1
    COBYLA = 2

integratedStateOffset = 0.0
def prepareIntegratingStateForPolisher(outputHist, outputHistReactorModel):
    global integratedStateOffset

    numberOfSamples = 20
    Ki = 1
    commonTime = outputHistReactorModel.time

    reactorC3 = interp1d(outputHist.time, outputHist.Cout[2, :], 'linear')(commonTime)
    modelC3 = outputHistReactorModel.Cout[2, :]

    if len(commonTime) < numberOfSamples: return False

    delta = (reactorC3 - modelC3 - integratedStateOffset)
    deltaWithinValidRegion = delta[-numberOfSamples:]
    integratedStateOffset += Ki * np.average(deltaWithinValidRegion, weights=np.linspace(1, 10, len(deltaWithinValidRegion)))

    integratedStateOffset = min(max(integratedStateOffset, -0.2), 0.2) 
    #if np.abs(integratedStateOffset) < 1e-4: integratedStateOffset = 0.0

    return True

integratorMap = None
def prepareIntegratingStateForPolisherUsingIntegratorMap(outputHist, outputHistReactorModel, C3ref, currentC3RefValue, interpolateAnchorPoints=False):
    global integratorMap, integratedStateOffset

    numberOfSamplePoints = 5
    if integratorMap is None:
        integratorMap = np.nan * np.zeros((numberOfSamplePoints))  # C2 Ref

    anchorPointsC3Ref = np.linspace(0, 1, numberOfSamplePoints)
    idxC3AnchorPoint = np.argmin(np.abs(anchorPointsC3Ref - C3ref))

    print(f"\t\tMap Entry: {idxC3AnchorPoint} --> {integratorMap[idxC3AnchorPoint]}")

    if not np.isnan(integratorMap[idxC3AnchorPoint]):
        integratedStateOffset = integratorMap[idxC3AnchorPoint]
    else:
        #if interpolateAnchorPoints and not np.all(np.isnan(integratorMap)):
        if interpolateAnchorPoints and np.sum(~np.isnan(integratorMap)) > 1:
            integratedStateOffset = np.interp(C3ref, anchorPointsC3Ref[~np.isnan(integratorMap)], integratorMap[~np.isnan(integratorMap)])
        else:
            integratedStateOffset = 0.0

    # Only Integrate and change state if we are in the area of the reference
    # Before take last integrated state offset
    if np.abs(C3ref - currentC3RefValue) < 0.05:
        print(f"\t\tDo integration - Ref to use within current Cref")
        stateChanged = prepareIntegratingStateForPolisher(outputHist, outputHistReactorModel)
    else:
        print(f"\t\tDo NOT integrate - Ref to use not within current Cref")
        stateChanged = True

    integratorMap[idxC3AnchorPoint] = integratedStateOffset

    return stateChanged



def resetIntegratingStateForPolisher():
    global integratedStateOffset
    integratedStateOffset = 0.0

def getIntegratedStateOffset():
    global integratedStateOffset
    return integratedStateOffset   


def pbmBasedCostFunction(x, reactorX0Data, C3ref, simTime, flowRate, temperature, plotFlag=False, outputHist=None, inputSignalSampleCount=10):
    # if outputHist is not None >> using integrating state or const offset at output
    reactorModel = getReactorModel()
    reactorModel.setLastSimResult(reactorX0Data)

    simTimeStart = 0 # anyway relative
    timeVec = np.linspace(simTimeStart, simTimeStart + simTime, inputSignalSampleCount)

    Cin = np.zeros((4, len(timeVec)))
    Cin[0, :] = x[0]                            # C1
    Cin[1, :] = x[1]                            # C2    
    flowRateVec = flowRate * np.ones_like(timeVec)     # flowRate
    temperatureVec = temperature * np.ones_like(timeVec)  # temperature

    time, Cout, _ = reactorModel.simulateStep(
        simTimeStart, simTimeStart + simTime,
        timeVec, Cin, flowRateVec, temperatureVec
    )

    # Tackle measurement uncertainties
    Cout[2, :] += integratedStateOffset

    # Calculate error over prediction vs. reference
    ##interpPredictionCout = interp1d(time, Cout)(timeVec[1:]) 
    # Take only the last third
    deltaRef = Cout[2, -len(timeVec)//3:] - C3ref
    error = deltaRef**2
    totalError = np.average(error)
    
    economicCost = x[0] + 10 * x[1] + (temperature / 3000)

    alteredCost = 0
    #if prevXopt is not None:
    #    costRelToPrevInput = lambda data_, label_: np.sqrt(np.sum((data_ - prevXopt.get(label_))**2)) / len(Cin[0, :])
    #    alteredCost = costRelToPrevInput(Cin[0, :], "C1") + costRelToPrevInput(Cin[1, :], "C2") + costRelToPrevInput(flowRate, "flowRate") + costRelToPrevInput(temperature, "temp")

    if plotFlag: reactorModel.plot()

    cost = 1e4 * totalError # + economicCost + 50 * alteredCost
    return cost

def polishOptimum(mode, polisher, sentinel, optimum, currentTime, inputHist, deltaTsim, C3RefValue, MinMaxC1C2=(0,1), MinMaxFlowRate=(0.2,2), MinMaxTemp=(50,200), outputHist=None):
    timeBackForPrevsimulation = 10*60

    prevTimeVec = np.arange(currentTime-timeBackForPrevsimulation, currentTime + (deltaTsim / 2), deltaTsim)
    prevCin = inputHist.getCinForTimeVec(prevTimeVec)
    prevFlowRate = inputHist.getFlowRateForTimeVec(prevTimeVec)
    prevTemperature = inputHist.getTemperatureForTimeVec(prevTimeVec)

    # Generate lastSimResult using reactorModel
    # "What happened so far"
    reactorModel = getReactorModel(getModelForOptimization=True)
    reactorModel.simulate(prevTimeVec, prevCin, prevFlowRate, prevTemperature)
    reactorX0Data = reactorModel.getLastSimResult()

    x0 = [optimum.get("C1"), optimum.get("C2"), optimum.get("flowRate"), optimum.get("temp")]
    bounds = [(np.max([xi - .1*xi, bbi[0]]), np.min([xi + .1*xi, bbi[1]])) for (xi, bbi) in zip(x0, [MinMaxC1C2, MinMaxC1C2, MinMaxFlowRate, MinMaxTemp])]

    constraints = []
    simTime = 10*60


    match mode:
        case MODE.C1C2: # optimize C1, C2 only
            x0 = x0[0:2]
            bounds = bounds[0:2]
            flowRate = optimum.get("flowRate")
            temperature = optimum.get("temp")
        case MODE.C1C2FlowRate: # optimize C1, C2, flowRate
            raise NotImplementedError("Not implemented yet")
        case MODE.C1C2Temp: # optimize C1, C2, temperature
            raise NotImplementedError("Not implemented yet")
        

    costFunction = pbmBasedCostFunction
    costFunctionArgs = (reactorX0Data, C3RefValue, simTime, flowRate, temperature, False)

    if sentinel == SENTINEL.MPC_INTEGRATING_PBM:
        if outputHist is None: raise ValueError("outputHist must be provided when using MPC_INTEGRATING_PBM sentinel")
        costFunction = pbmBasedCostFunction
        costFunctionArgs = (reactorX0Data, C3RefValue, simTime, flowRate, temperature, False, outputHist)
    else:
        # Ensure to reset the integrated state offset
        global integratedStateOffset
        integratedStateOffset = 0.0



    match polisher:
        case POLISHER.NONE:
            polishedOptimum = {
                "C1" : x0[0], "C2" : x0[1], "temp": temperature, "flowRate": flowRate,
                "cost": optimum.get("cost"),
                "nfev": 0, "delta_C1": 0, "delta_C2": 0,
                "success": True,
            }
        case POLISHER.SLSQP:
            result = minimize(
                costFunction,
                x0,
                args=costFunctionArgs,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'ftol': 1e-10,
                    'disp': True,
                    'maxiter': 5000,
                    #'maxfunccalls': 1000,
                }
            )

            polishedOptimum = {
                "C1" : result.x[0], "C2" : result.x[1], "temp": temperature, "flowRate": flowRate,
                "cost": result.fun, "nfev": result.nfev, "delta_C1": result.x[0] - x0[0], "delta_C2": result.x[1] - x0[1],
                "success": result.success,
            }

        case POLISHER.COBYLA:
            result = minimize(
                costFunction,
                x0,
                args=costFunctionArgs,
                method='COBYLA',
                bounds=bounds,
                constraints=constraints,
                options={
                    'disp': False,
                    'maxiter': 5000,
                    #'maxfunccalls': 1000,
                }
            )

            polishedOptimum = {
                "C1" : result.x[0], "C2" : result.x[1], "temp": temperature, "flowRate": flowRate,
                "cost": result.fun, "nfev": result.nfev, "delta_C1": result.x[0] - x0[0], "delta_C2": result.x[1] - x0[1],
                "success": result.success,
            }



    return polishedOptimum
