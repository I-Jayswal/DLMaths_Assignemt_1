import pytest
import torch

def create_tensor_of_val(dimensions, val):
    return torch.full(dimensions, val)

def calculate_elementwise_product(A, B):
    return A * B

def calculate_matrix_product(X, W):
    return torch.matmul(X, W.T)

def calculate_matrix_prod_with_bias(X, W, b):
    sum_total = torch.matmul(X, W.T) + b  # Add bias here
    return sum_total

def calculate_activation(sum_total):
    return torch.heaviside(sum_total, torch.tensor(0.0, dtype=sum_total.dtype)).float()

def calculate_output(X, W, b):
    sum_total = calculate_matrix_prod_with_bias(X, W, b)
    return calculate_activation(sum_total)




