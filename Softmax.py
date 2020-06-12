import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

class SoftmaxNet(nn.Module):
    def __init__(self, in_features: int, out_labels: int, verbose=False):
        super(SoftmaxNet, self).__init__()
        self.verbose = verbose
        self.training = None

        self.fc1 = nn.Linear(in_features, out_labels)

        self.criterion = None
        self.optimizer = None
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.reshape(-1, self.fc1.in_features) if len(X) > 2 else X
        return self.fc1(X)
    
    def initialized(self, criterion=None, optimizer=None, learning_rate=0.1) -> None:
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        if optimizer is None:
            self.optimizer = SGD(self.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        self.fc1.weight.data.uniform_(0.0, 0.01)
        self.fc1.bias.data.fill_(0)

    def train(self, data: DataLoader, epochs: int, mode=True) -> None:
        if self.verbose:
            running_loss = 0.0
        
        self.training = mode
        for module in self.children():
            module.train(mode)

        for epoch in range(0, epochs):
            for i, datum in enumerate(data, 0):
                features, labels = datum
                loss = self.criterion(self(features), labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.verbose:
                    running_loss += loss.item()
                    batch_split = int(len(data.dataset) / data.batch_size / 5)
                    batch_split = 1 if batch_split < 1 else batch_split
                    if i % batch_split == batch_split-1:
                        print(f"[epoch {epoch+1}, batch {i+1}] loss: {running_loss/batch_split}")
                        running_loss = 0.0

        if self.verbose:
            print('Finished Training')
        return self
