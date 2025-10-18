
from NN import FullyConnectedNN

import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from Normalization import MinMaxNormalizer

from NN import NNWithContext


def main():
    NNDumpFilename="./NNWithContext.nn"

    myNN = NNWithContext(
        networkType=FullyConnectedNN,
        hidden_sizes=[200, 200, 200],
        timeBack=20*60, timePrediction=5*60, trainingSamplesPerSimulation=5, 
        predictSpecies=[2], includeCoutSpecies=[2], inputSpecies=[0, 1],
        sampleTimeNNInput=30,
        normalizer=MinMaxNormalizer()
    )

    #myNN = NNWithContext.LOAD(NNDumpFilename)

    X, Y, simulations = myNN.loadTrainingData([
            "./simulation_ConstAndRamps.pkl", 
            #"./small_simulation_ConstAndRamps.spkl"
        ],
        initNormalizerWithLoadedData=True
    )

    myNN.trainNN(
        X, Y,
        epochs=10000, epochMod=1, learning_rate=1e-3,
        dumpFilename=NNDumpFilename
    )
   
    
if __name__ == "__main__":
    main()