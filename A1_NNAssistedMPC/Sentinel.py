from Polisher import polishOptimum, prepareIntegratingStateForPolisher, getIntegratedStateOffset, prepareIntegratingStateForPolisherUsingIntegratorMap
from Common import SENTINEL

import numpy as np

def polishOptWithMPCIntegratingSentinel(mode, polisher, sentinel, opt, currentTime, inputHist, deltaTsim, C3RefValue, MinMaxC1C2, MinMaxFlowRate, MinMaxTemp, outputHist, outputHistReactorModel, useIntegratorMap=False, currentC3RefValue=None):
    print(f"\tMPC Integrating PBM Sentinel - within Polisher ({polisher.name})")

    if useIntegratorMap:
        integralStateChanged = prepareIntegratingStateForPolisher(outputHist, outputHistReactorModel)

    if integralStateChanged:
        polishedOpt = polishOptimum(mode, polisher, sentinel, opt, currentTime, inputHist, deltaTsim, C3RefValue, MinMaxC1C2, MinMaxFlowRate, MinMaxTemp, outputHist)
    else:
        print("\tIntegrated state offset did not change - do not re-polish")

    return polishedOpt