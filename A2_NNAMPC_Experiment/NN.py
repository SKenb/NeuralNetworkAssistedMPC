#import torch.nn as nn
import pickle
import torch.nn as nn
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split



class NNWithContext():
    def __init__(self, networkType, hidden_sizes, timeBack, timePrediction, trainingSamplesPerSimulation, predictSpecies, sampleTimeNNInput=30, includeCoutSpecies=[], inputSpecies=[0, 1], normalizer=None, test_size=0.2):
        self.networkType = networkType
        self.hidden_sizes = hidden_sizes

        self.timeBack = timeBack
        self.timePrediction = timePrediction
        self.trainingSamplesPerSimulation = trainingSamplesPerSimulation
        self.sampleTimeNNInput = sampleTimeNNInput
        self.includeCoutSpecies = includeCoutSpecies
        self.predictSpecies = predictSpecies
        self.inputSpecies = inputSpecies

        self.normalizer = normalizer
        self.test_size = test_size

        input_size, output_size = self._getNNInputOutputSizes()
        self.NeuralNetwork = networkType(input_size, hidden_sizes, output_size)


    def _getNNInputOutputSizes(self):
        numberOfInputs = len(self.inputSpecies) + len(self.includeCoutSpecies) + len(self.predictSpecies) + 2 # flow rate and temperature
        
        input_size = self.timeBack // self.sampleTimeNNInput * numberOfInputs
        output_size = self.timePrediction // self.sampleTimeNNInput * len(self.predictSpecies)

        return input_size, output_size

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def LOAD(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
    def loadTrainingData(self, filenames, initNormalizerWithLoadedData=False, verbose=True):          
        raise NotImplementedError("Loading training data from files is not implemented yet IN THIS VERSION ONLY.")
    
    def _splitAndPrepareData(self, X, Y):
        X = torch.from_numpy(X).to(torch.float)
        Y = torch.from_numpy(Y).to(torch.float)

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=self.test_size, random_state=42)
        return X_train, X_val, Y_train, Y_val
    
    def predictData(self, X):
        X_norm, _ = self.normalizer.normalize(X) if self.normalizer is not None else (X, None)
        
        if isinstance(X_norm, torch.Tensor):
            prediction = self.NeuralNetwork(X_norm.to(torch.float))
        else:
            prediction = self.NeuralNetwork(torch.from_numpy(X_norm).to(torch.float)).detach().numpy()
        
        _, prediction = self.normalizer.denormalize(Y=prediction) if self.normalizer is not None else (None, prediction)

        return prediction


class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FullyConnectedNN, self).__init__()
        layers = []

        # Create input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.Sigmoid())

        # Create hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.Sigmoid())

        # Create output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Use nn.Sequential to stack the layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    def optimize_u(self, u_init, target, prepare_input_fn, lr=0.01, n_iter=100):
        u = torch.tensor(u_init, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([u], lr=lr)

        for _ in range(n_iter):
            optimizer.zero_grad()
            X = prepare_input_fn(u)           # shape (N, input_size)
            pred = self.forward(X).squeeze()  # shape (N,)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            optimizer.step()

        return u.detach()