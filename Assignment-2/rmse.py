import numpy as np
import pytest

def rmse(predictions, targets):
    # Convert the inputs to NumPy arrays
    pred = np.array(predictions)
    tar = np.array(targets)

    # Calculate the RMSE
    rmse = np.sqrt(np.mean((pred - tar) ** 2))

    return rmse

