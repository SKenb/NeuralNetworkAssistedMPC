import numpy as np
from enum import Enum
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d 

from Common import getReactorModel

class POLISHER(Enum):
    NONE = 0
    SLSQP = 1
    COBYLA = 2

#integratedStateOffset = 0.0
def prepareIntegratingStateForPolisher(integratedStateOffset, outputHist, outputHistReactorModel):
    #global integratedStateOffset

    numberOfSamples = 6
    Ki = .5
    commonTime = outputHistReactorModel.time

    reactorC3 = interp1d(outputHist.time, outputHist.Cout[2, :], 'linear')(commonTime)
    modelC3 = outputHistReactorModel.Cout[2, :]

    if len(commonTime) < numberOfSamples: return False, integratedStateOffset

    print(integratedStateOffset)
    print(modelC3)
    print(reactorC3)
    
    delta = (reactorC3 - modelC3 - integratedStateOffset)
    deltaWithinValidRegion = delta[-numberOfSamples:]
    integratedStateOffset += Ki * np.average(deltaWithinValidRegion, weights=np.linspace(1, numberOfSamples, len(deltaWithinValidRegion)))

    integratedStateOffset = min(max(integratedStateOffset, -0.1), 0.1) 
    #if np.abs(integratedStateOffset) < 1e-4: integratedStateOffset = 0.0

    return True, integratedStateOffset

def pbmBasedCostFunction(x, print, reactorX0Data, C3ref, integratedStateOffset, simTime, flowRate, temperature, plotFlag=False, outputHist=None, inputSignalSampleCount=10):
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

    print("\t\t\t\t\t>> Sim within optimization step")
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

def polishOptimum(print, optimum, currentTime, inputHist, deltaTsim, C3RefValue, integratedStateOffset, outputHist, MinMaxC1C2=(0,1), MinMaxFlowRate=(0.2,2), MinMaxTemp=(50,200)):
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

    #constraints = []
    simTime = 10*60

    constraints = [
        {"type": "ineq", "fun": lambda x: x[0]},            # x[0] >= 0
        {"type": "ineq", "fun": lambda x: x[1]},            # x[1] >= 0
    ]

    x0 = x0[0:2]
    bounds = bounds[0:2]
    flowRate = optimum.get("flowRate")
    temperature = optimum.get("temp")
        
    costFunction = pbmBasedCostFunction
    costFunctionArgs = (print, reactorX0Data, C3RefValue, integratedStateOffset, simTime, flowRate, temperature, False, outputHist)

    print("\t\t\t\t>> Mimimizing cost function using COBYLA")
    print("\t\t\t\t>> Initial guess: C1: {:.3f}, C2: {:.3f}, FlowRate: {:.3f}, Temp: {:.1f}°C, IntegratedStateOffset: {:.3f}".format(x0[0], x0[1], flowRate, temperature, integratedStateOffset))
    result = minimize(
        costFunction,
        x0,
        args=costFunctionArgs,
        method='COBYLA',
        #bounds=bounds, <-- doesn't take bounds
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
