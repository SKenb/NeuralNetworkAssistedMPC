import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from Container import SimulationInstances

import pickle


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

    def LOAD(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
    def loadTrainingData(self, filenames, initNormalizerWithLoadedData=False, verbose=True):    
               
        simulations = SimulationInstances("All simulations", None)

        for filename in filenames:
            simulations.load(filename)  

        if verbose: print(f"Loaded simulations ({len(simulations.instances)} simulations)")

        X, Y = simulations.getTrainingData(
            self.timeBack, self.timePrediction, self.sampleTimeNNInput, 
            self.trainingSamplesPerSimulation, self.includeCoutSpecies, self.predictSpecies
        )

        if verbose: 
            print(f"We got {X.shape[0]} training sets with {X.shape[1]} inputs and {Y.shape[1]} outputs.")
            print(f"\tWe take the last {(self.timeBack/60):.2f} minutes to estimate the next {(self.timePrediction/60):.2f} minutes.")

        if initNormalizerWithLoadedData and self.normalizer is not None:
            self.normalizer.initWithData(X, Y)

        return X, Y, simulations
    
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
    
    def trainNN(self, X, Y, learning_rate=1e-3, epochs=1e3, epochMod=100, dumpFilename=None, batch_size=320):
        X_train, X_val, Y_train, Y_val = self._splitAndPrepareData(X, Y)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.NeuralNetwork.parameters(), lr=learning_rate)

        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and Y_val is not None:
            val_dataset = TensorDataset(X_val, Y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(int(epochs)):
            model_dumped = "NO"
            epoch_loss = 0.0

            self.NeuralNetwork.train() # activates dropout handling
            #outputs = model(X_train)
            #loss = criterion(outputs, Y_train)

            # Loop through batches
            for batch_X, batch_Y in train_loader:
                # Normalize the batch if a normalizer is provided
                if self.normalizer is not None:
                    batch_X, batch_Y = self.normalizer.normalize(batch_X.numpy(), batch_Y.numpy())
                    batch_X = torch.from_numpy(batch_X).to(torch.float)
                    batch_Y = torch.from_numpy(batch_Y).to(torch.float)

                # Forward pass
                outputs = self.NeuralNetwork(batch_X)
                loss = criterion(outputs, batch_Y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss for the epoch
                epoch_loss += loss.item()

            if (epoch + 1) % epochMod == 0:
                # Save model
                if dumpFilename is not None:
                    try:
                        #torch.save(model.state_dict(), dumpFilename) # model.load_state_dict(torch.load(dumpFilename))
                        self.save(dumpFilename)
                        model_dumped = "YES"
                    except Exception as e:
                        print(f"Error saving model: {e}")

                # Validation step
                if X_val is not None and Y_val is not None:
                    self.NeuralNetwork.eval()  # Set model to evaluation mode - stops dropout
                    val_loss = 0.0

                    with torch.no_grad():  # Disable gradient computation
                        for val_X, val_Y in val_loader:

                            # Normalize the validation batch if a normalizer is provided
                            if self.normalizer is not None:
                                val_X, val_Y = self.normalizer.normalize(val_X.numpy(), val_Y.numpy())
                                val_X = torch.from_numpy(val_X).to(torch.float)
                                val_Y = torch.from_numpy(val_Y).to(torch.float)

                            val_outputs = self.NeuralNetwork(val_X)
                            val_loss += criterion(val_outputs, val_Y).item()

                    print(f'Epoch [{epoch + 1}/{epochs}],\tTrain Loss: {epoch_loss / len(train_loader):.4f},\tValidation MSE: {val_loss / len(val_loader):.4f},\tDumped: {model_dumped},\tDetailed EL:{epoch_loss*1e6:.4f}')
                else:
                    print(f'Epoch [{epoch + 1}/{epochs}],\tTrain Loss: {epoch_loss / len(train_loader):.4f},\tDumped: {model_dumped},  Detailed EL:{epoch_loss*1e6:.4f}')




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
        """
        Optimizes control signal u to minimize MSE between NN(u) and target.

        Args:
            u_init (np.ndarray or torch.Tensor): shape (4, N)
            target (torch.Tensor): shape (N,) or (batch,)
            prepare_input_fn: function to prepare NN input from u
            lr: learning rate
            n_iter: number of iterations
        """
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