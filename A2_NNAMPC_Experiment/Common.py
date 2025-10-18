from FlowReactor import FlowReactor
from Container import OutputHist, InputHist
from Reaction import paalKnorrReaction
from enum import Enum
import time

def parseInputs(dataFromOptipus):
    inputData = dataFromOptipus.get("inputs", None)
    if inputData is None: raise ValueError("Missing 'data' in inputs")
    
    C3meas = inputData.get("C3meas", None)
    C3ref = inputData.get("C3ref", None)
    temperature = inputData.get("temperature", None)
    flowRate = inputData.get("flowRate", None)

    if C3meas is None: raise ValueError("Missing 'C3meas' in inputs")
    if C3ref is None: raise ValueError("Missing 'C3ref' in inputs")
    if temperature is None: raise ValueError("Missing 'temperature' in inputs")
    if flowRate is None: raise ValueError("Missing 'flowRate' in inputs")

    return C3meas, C3ref, temperature, flowRate

def dataForOptipus(C1, C2):
    return {
        "Cin1": C1,
        "Cin2": C2,
    }

def remember(timeOffset, prevC3ref, inputHist, outputHist, integratedStateOffset, dse_optimum, iteration):
    return {
        "timeOffset": timeOffset,
        "prevC3ref": prevC3ref,
        "inputHist": inputHist,
        "outputHist": outputHist,
        "dse_optimum": dse_optimum,
        "integratedStateOffset": integratedStateOffset,
        "iteration": iteration
    }


def tryToRemember(dataFromOptipus):
    remember = dataFromOptipus.get("remember", None)
    if remember is None: return None, None, InputHist(), OutputHist(), 0, None, 1

    timeOffset = remember.get("timeOffset", None)
    prevC3ref = remember.get("prevC3ref", None)

    outputHist = remember.get("outputHist", OutputHist())
    inputHist = remember.get("inputHist", InputHist())
    
    integratedStateOffset = remember.get("integratedStateOffset", 0)
    dse_optimum = remember.get("dse_optimum", None)
    iteration = remember.get("iteration", 0) + 1

    return timeOffset, prevC3ref, inputHist, outputHist, integratedStateOffset, dse_optimum, iteration


reactorModel = None
reactorModelForOptimization = None
def getReactorModel(getModelForOptimization=True):
    # Can/Should be used for PBM base optimization
    # Parameters where altered to simulate model uncertainties
    global reactorModel, reactorModelForOptimization

    pKR = lambda Cs, i, theta: paalKnorrReaction(
        Cs, i, theta, 
        A1=0.0548, A2=7.2275e-04, Ea1=3.3889e+03, Ea2=2.0240, k11=0.7502, k12=0.8701, k21=0.7003, k22=1.2938 
    )

    length = 8.41
    diameter = 0.8*1e-3
    dispersionCoefficient = 2e-4
    setSpaceSamples = 100

    if getModelForOptimization:
        if reactorModelForOptimization is not None: return reactorModelForOptimization

        reactorModelForOptimization = FlowReactor(
            length=length, diameter=diameter, 
            dispersionCoefficient=dispersionCoefficient, 
            reactionNetworkCallback=pKR,
            setSpaceSamples=setSpaceSamples
        )
        
        return reactorModelForOptimization
    else:
        if reactorModel is not None: return reactorModel
        
        reactorModel = FlowReactor(
            length=length, diameter=diameter, 
            dispersionCoefficient=dispersionCoefficient, 
            reactionNetworkCallback=pKR,
            setSpaceSamples=setSpaceSamples
        )
        
        return reactorModel
