from Polisher import polishOptimum, prepareIntegratingStateForPolisher
import numpy as np

def polishOptWithMPCIntegratingSentinel(print, optimum, currentTime, inputHist, integratedStateOffset, deltaTsim, C3RefValue, MinMaxC1C2, MinMaxFlowRate, MinMaxTemp, outputHist, outputHistReactorModel, forcePolishing=False):
    print(f"\t\t\tMPC Integrating PBM Sentinel ")

    integralStateChanged, integratedStateOffset = prepareIntegratingStateForPolisher(integratedStateOffset, outputHist, outputHistReactorModel)

    if integralStateChanged or forcePolishing:
        polishedOpt = polishOptimum(print, optimum, currentTime, inputHist, deltaTsim, C3RefValue, integratedStateOffset, outputHist, MinMaxC1C2, MinMaxFlowRate, MinMaxTemp)
    else:
        print("\t\t\t>> Integrated state offset did not change - do not re-polish")
        polishedOpt = optimum

    return polishedOpt, integratedStateOffset