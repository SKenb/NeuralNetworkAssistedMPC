import pickle
import numpy as np
import torch

class Normalizer:
    def __init__(self, name):
        self.name = name

        
    def initWithData(self, X, Y):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def normalize(self, X, Y):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def denormalize(self, X, Y):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def LOAD(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    
class MinMaxNormalizer(Normalizer):
    def __init__(self):
        super().__init__("MinMax")
        self.X_min = None
        self.X_max = None

    def initWithData(self, X, Y):
        self.X_min = np.min(X, axis=0)
        self.X_max = np.max(X, axis=0)

        self.Y_min = np.min(Y, axis=0)
        self.Y_max = np.max(Y, axis=0)

    def normalize(self, X=None, Y=None):
        if X is None and Y is None: return None, None

        X_max, X_min = self.X_max, self.X_min
        Y_max, Y_min = self.Y_max, self.Y_min

        if isinstance(X, torch.Tensor):
            X_max, X_min = torch.from_numpy(self.X_max), torch.from_numpy(self.X_min)
            Y_max, Y_min = torch.from_numpy(self.Y_max), torch.from_numpy(self.Y_min)

        range_X = X_max - X_min
        range_X[range_X == 0] = 1.0
        X_normalized = (X-X_min) / range_X
        if Y is None: return X_normalized, None

        range_Y = Y_max - Y_min
        range_Y[range_Y == 0] = 1.0
        Y_normalized = (Y-Y_min) / range_Y

        return X_normalized, Y_normalized
    
    def denormalize(self, X=None, Y=None):
        if X is None and Y is None: return None, None

        X_max, X_min = self.X_max, self.X_min
        Y_max, Y_min = self.Y_max, self.Y_min

        if isinstance(Y, torch.Tensor):
            X_max, X_min = torch.from_numpy(self.X_max), torch.from_numpy(self.X_min)
            Y_max, Y_min = torch.from_numpy(self.Y_max), torch.from_numpy(self.Y_min)

        range_Y = Y_max - Y_min
        range_Y[range_Y == 0] = 1.0
        Y_denormalized = Y * range_Y + Y_min
        if X is None: return None, Y_denormalized

        range_X = X_max - X_min
        range_X[range_X == 0] = 1.0
        X_denormalized = X * range_X + X_min

        return X_denormalized, Y_denormalized