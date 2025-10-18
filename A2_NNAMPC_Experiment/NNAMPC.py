import numpy as np

from NN import NNWithContext
from scipy.interpolate import interp1d

from DesignSpaceExploration import designSpaceExploration_fixedTemperatureAndFixedFlowRate, getRegionAroundOptimum, findMinimumCost
from Common import getReactorModel
from Polisher import polishOptimum
from Sentinel import polishOptWithMPCIntegratingSentinel

def NN_A_MPC(print, currentTime, C3ref, temperature, flowRate, prevC3ref, inputHist, outputHist, integratedStateOffset, outputHistReactorModel, dse_optimum, iteration):

    nnModel = NNWithContext.LOAD(".\\NNWithContextOptParams.nn")

    sampleTimeMPC = 10*60
    sampleTimeSentinel = 2*60
    # --> each 5 time steps of the MPC, one sentinel step
    doDSE = (iteration - 1) % (sampleTimeMPC // sampleTimeSentinel) == 0
    
    predictionHorizon = 30*60

    # Design space exploration
    if doDSE or dse_optimum is None or prevC3ref is None or np.abs(prevC3ref - C3ref) > .1:
         print("\t\tDesign space exploration step")
         dse_optimum, _ = runDesignSpaceExplorationStep(print, currentTime, predictionHorizon, C3ref, prevC3ref, nnModel, temperature, flowRate, outputHist, dse_optimum)
     
    # Sentinel
    forcePolishing = doDSE or dse_optimum is None
    print(f"\t\tPolishing with PBM + integrating state ({'forced' if forcePolishing else 'if changed'})")
    C1, C2, integratedStateOffset = runSentinelStep(print, currentTime, dse_optimum, integratedStateOffset, C3ref, flowRate, temperature, inputHist, outputHist, outputHistReactorModel, forcePolishing)
    
    return C1, C2, dse_optimum, integratedStateOffset

def runSentinelStep(print, currentTime, optimum, integratedStateOffset, C3ref, flowRate, temperature, inputHist, outputHist, outputHistReactorModel, forcePolishing):
    deltaTsim = 30
    MinMaxC1C2 = (0,1)
    MinMaxFlowRate = (flowRate, flowRate)
    MinMaxTemp = (temperature, temperature)
   
    polishedOpt, integratedStateOffset = polishOptWithMPCIntegratingSentinel(print, optimum, currentTime, inputHist, integratedStateOffset, deltaTsim, C3ref, MinMaxC1C2, MinMaxFlowRate, MinMaxTemp, outputHist, outputHistReactorModel, forcePolishing)

    C1, C2 = polishedOpt.get("C1"), polishedOpt.get("C2")
    print(f"\tPolished optimum (using measured data): C1: {C1:.3f}\tC2: {C2:.3f}")

    return C1, C2, integratedStateOffset

def runPolishingStep(print, optimum, currentTime, C3ref, temperature, flowRate, inputHist, outputHist):
    deltaTsim = 30
    MinMaxC1C2 = (0, .6)
    MinMaxFlowRate = (flowRate, flowRate)
    MinMaxTemp = (temperature, temperature)

    integratedStateOffset = 0
    
    polishedOpt = polishOptimum(print, optimum, currentTime, inputHist, deltaTsim, C3ref, integratedStateOffset, outputHist, MinMaxC1C2, MinMaxFlowRate, MinMaxTemp)

    C1, C2 = polishedOpt.get("C1"), polishedOpt.get("C2")
    #flowRate = polishedOpt.get("flowRate") * np.ones_like(timeVec)
    #temperature = polishedOpt.get("temp") * np.ones_like(timeVec)

    print(f"\t\t\tPolished optimum: C1: {C1:.3f}\tC2: {C2:.3f}")
    return C1, C2

def runDesignSpaceExplorationStep(print, currentTime, predictionHorizon, C3ref, prevC3ref, myNN, temperature, flowRate, outputHist, dse_optimum):

    if prevC3ref is None: prevC3ref = C3ref + 10 # ensure first time global search
    deltaTsim = 2*myNN.sampleTimeNNInput

    predTimeVec = np.arange(currentTime, currentTime+predictionHorizon+deltaTsim, deltaTsim)
    print(f"\t\t\tTime: {currentTime/60} min ({currentTime}s)\tPredict until: {predTimeVec[-1]/60}min")

    ## Do the optimization
    Cin1Bounds, Cin2Bounds = (0, 1), (0, 1)
    flowRateBounds = (0.2, 2)
    temperatureBounds = (50, 200)
        
    if np.abs(prevC3ref - C3ref) > .1 or not dse_optimum: # if True:
        print(f"\t\t\t>> Starting optimization - global search")
        # NEW Cref - do global search
        samples = 5
        overlap = 20
        rangeSizeInPercentage = 50
        maxIterations = 8
        includeCostUsingPrevXOpt = False

    else:
        print(f"\t\t\t>> Starting optimization - around previous optimum")
        aroundOldBound = lambda oldBounds_, border_, min_, max_: (max((oldBounds_[0] - border_), min_), min((oldBounds_[1] + border_), max_)) 
        # Cref quite similar - look in sourounding
        Cin1Bounds = aroundOldBound(Cin1Bounds, .2, 0, 1)
        Cin2Bounds = aroundOldBound(Cin2Bounds, .2, 0, 1)
        flowRateBounds = aroundOldBound(flowRateBounds, .4, 0.2, 2)
        temperatureBounds = aroundOldBound(temperatureBounds, 50, 50, 200)
        samples = 5
        overlap = 40
        rangeSizeInPercentage = 80
        maxIterations = 4
        includeCostUsingPrevXOpt = True

    results = []
    overlapHist = []
    rangeSizeInPercentageHist = []
    xOptMoveHist = []
    prevXopt = None

    boundsDelta = lambda bounds_: bounds_[1] - bounds_[0]
    distanceBreakCritera = False

    while (boundsDelta(Cin1Bounds) > .1 or boundsDelta(Cin2Bounds) > .1) and maxIterations >= len(overlapHist) and not distanceBreakCritera:
        overlapHist.append(overlap)
        rangeSizeInPercentageHist.append(rangeSizeInPercentage)

        result = designSpaceExploration_fixedTemperatureAndFixedFlowRate(myNN, outputHist, predTimeVec, C3ref, overlap, temperature, flowRate, Cin1Bounds, Cin2Bounds, samples, verbose=False, prevXopt=dse_optimum if includeCostUsingPrevXOpt and not dse_optimum else None)
        results.append(result)
    
        print(f"\t\t\t\t>> dSE - C1: {Cin1Bounds}\tC2: {Cin2Bounds}\tOverlap: {overlap}%\tRange: {rangeSizeInPercentage}%")

        newRanges = getRegionAroundOptimum(results[-1], rangeSizeInPercentage)

        Cin1Bounds, Cin2Bounds = newRanges.get("newC1Range"), newRanges.get("newC2Range")
        #flowRateBounds = newRanges.get("newFlowRateRange")
        overlap += 10 #20, 10 <-- faster
        overlap = min(overlap, 100)
        rangeSizeInPercentage *= .75 if rangeSizeInPercentage >= 10 else .5

        opt = findMinimumCost(result)
        Xopt = np.array([opt.get("C1"), opt.get("C2"), opt.get("temp"), opt.get("flowRate")])
        if prevXopt is not None: xOptMoveHist.append(np.sqrt(np.sum((prevXopt - Xopt)**2)))
        prevXopt = Xopt

        distanceBreakCritera = len(xOptMoveHist) > 4 and np.sum(xOptMoveHist[-1:-5:-1])/5 < 0.05

    opt = findMinimumCost(results[-1])
    print(f"\t\t\tOptimization finished - Best cost: {opt.get('cost')} at C1: {opt.get('C1')}, C2: {opt.get('C2')}")
    return opt, results[-1]