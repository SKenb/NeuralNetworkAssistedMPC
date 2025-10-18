from FlowReactor import FlowReactor
from Reaction import paalKnorrReaction
from enum import Enum

class MODE(Enum):
    C1C2 = 1
    C1C2FR = 2
    C1C2TEMP = 3

class SENTINEL(Enum):
    NONE = 0
    PI_BAND_C1 = 1
    MPC_INTEGRATING_PBM = 2

realReactor = None
def getRealReactor(useAlteredParameters=True):
    # Represents the real reactor
    # Parameters where optimized in FlowMat
    # Accurate model
    global realReactor
    if realReactor is not None: return realReactor

    if useAlteredParameters:
        pKR = lambda Cs, i, theta: paalKnorrReaction(
            Cs, i, theta, 
            A1=0.055, A2=6.5e-04, Ea1=3e+03, Ea2=2, k11=0.75, k12=0.87, k21=0.7, k22=1.29 # ALTERED to simulate model uncertainties
        )
    else:
        pKR = lambda Cs, i, theta: paalKnorrReaction(
            Cs, i, theta, 
            A1=0.0548, A2=7.2275e-04, Ea1=3.3889e+03, Ea2=2.0240, k11=0.7502, k12=0.8701, k21=0.7003, k22=1.2938 # FROM optimization with k1 - k2
        )

    realReactor = FlowReactor(
        length=8.41, diameter=0.8*1e-3, 
        dispersionCoefficient=2e-4, 
        reactionNetworkCallback=pKR,
        setSpaceSamples=100
    )
    
    return realReactor

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
