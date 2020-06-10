import torch

def synthetic_data(w, b, num_examples):
    X = torch.zeros(size=(num_examples, w.shape[0])).normal_()
    y = X @ w + b
    y += torch.zeros(size=y.shape).normal_(std=0.01)
    return X, y