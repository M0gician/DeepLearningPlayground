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
    
    def initialize(self, criterion=None, optimizer=None, learning_rate=0.1) -> None:
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
                pred = self(features)
                loss = self.criterion(pred, labels)
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


if __name__ == "__main__":
    import torch
    from helper import load_fashion_mnist_data, predict_fashion_mnist
    from helper import evaluate_accuracy, evaluate_loss
    from torch.utils.data import DataLoader
    from Softmax import SoftmaxNet

    # Input size: 28 (width) * 28 (height) = 784 pixels
    num_inputs = 784

    # Output size: 10 categories
    num_outputs = 10

    batch_size = 256
    mnist_train, mnist_test = load_fashion_mnist_data(batch_size, resize=None)

    smNet = SoftmaxNet(num_inputs, num_outputs, verbose=True)
    smNet.initialize(learning_rate=0.15)
    smNet.train(mnist_train, 5)