import torch
import typing
import torch.nn as nn
from MultiLayerPerceptron import Sequential
from collections import OrderedDict
from typing import Union, TypeVar

T = TypeVar('T')
Layer_Type = Union[typing.OrderedDict[str, nn.Module], nn.Module]


class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


class Flatten(nn.Module):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.view(-1, int(torch.numel(X) / X.size(0)))


class LeNet(Sequential):
    def __init__(self, *layers: Layer_Type, verbose=False):
        super(LeNet, self).__init__()
        self.verbose = verbose
        self.training = False

        self.criterion = None
        self.optimizer = None

        if len(layers) == 1 and isinstance(layers[0], OrderedDict):
            for name, module in layers[0].items():
                self.add_module(name, module)
        else:
            for idx, module in enumerate(layers):
                self.add_module(str(idx), module)

    def train(self, mode=True, data=None, epochs=10) -> 'LeNet':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        self.to(device)

        if data is None:
            raise FileNotFoundError("\"data\" has to be a valid Dataloader object!")
        if self.verbose:
            running_loss = 0.0

        self.training = mode
        for module in self:
            module.train(mode)

        for epoch in range(0, epochs):
            for i, datum in enumerate(data, 0):
                features, labels = datum[0].to(device), datum[1].to(device)
                loss = self.criterion(self(features), labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.verbose:
                    running_loss += loss.item()
                    batch_split = int(len(data.dataset) / data.batch_size / 5)
                    batch_split = 1 if batch_split < 1 else batch_split
                    if i % batch_split == batch_split - 1:
                        print(f"[epoch {epoch + 1}, batch {i + 1}] loss: {running_loss / batch_split}")
                        running_loss = 0.0

        if self.verbose:
            print('Finished Training')
        return self


class LeNet_C3(nn.Module):
    """
    Implementing the original convolution C3 described in the paper
        "Gradient-Based Learning Applied to Document Recognition".
    
        │ 0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15
    ────┼─────────────────────────────────────────────────────────
      0 │ x           x  x  x        x  x   x   x       x   x
      1 │ x  x           x  x  x        x   x   x   x       x
      2 │ x  x  x           x  x  x         x       x   x   x
      3 │    x  x  x        x  x  x  x          x       x   x
      4 │       x  x  x        x  x  x  x       x   x       x
      5 │          x  x  x        x  x  x   x       x   x   x

       Each column indicates which feature map in S2 are combined 
             by the units in a particular feature map of C3
    """

    def __init__(self):
        super(LeNet_C3, self).__init__()
        self.connect_3 = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [0, 4, 5],
            [0, 1, 5],
        ]

        self.connect_4 = [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [1, 3, 4, 5],
            [1, 2, 4, 5],
            [1, 2, 3, 5],
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [0, 2, 3, 5]
        ]

        self.connect_6 = [0, 1, 2, 3, 4, 5]

        # Trainable parameters: (5*5*1)*3*6 + 1*6 = 456
        self.connect_3_modules = nn.ModuleList(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(5, 5), padding=False) for i in range(
                len(self.connect_3)
            )
        )

        # Trainable parameters: (5*5*1)*4*9 + 1*9 = 909
        self.connect_4_modules = nn.ModuleList(
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(5, 5), padding=False) for i in range(
                len(self.connect_4)
            )
        )

        # Trainable parameters: (5*5*1)*6*1 + 1*1 = 151
        self.connect_6_modules = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(5, 5), padding=False)

        # Total parameters: 456 + 909 + 151 = 1516

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        def calc_routine(subConv: nn.Conv2d, connection: list, input=X) -> torch.Tensor:
            return subConv(input[:, connection, :, :])

        connect_3_output = list(map(calc_routine, self.connect_3_modules, self.connect_3))
        connect_4_output = list(map(calc_routine, self.connect_4_modules, self.connect_4))
        connect_6_output = [self.connect_6_modules(X[:, self.connect_6, :, :])]
        return torch.cat(connect_3_output + connect_4_output + connect_6_output, dim=1)


class Subsampler(nn.Module):
    def __init__(self, in_channels: int):
        super(Subsampler, self).__init__()
        self.in_channels = in_channels

        self.sampling_module = nn.AvgPool2d(kernel_size=2, stride=2)
        self.trainable_module = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            groups=self.in_channels,
            bias=True
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.sampling_module(X)
        output = self.trainable_module(output)
        return output


"""
Layer C1 (Convolutional):
    Input size: 32x32
    Output size: 28x28
    Input channels: 1
    Output channels: 6
    Kernel size: 5x5
    Stride: 1
    Padding: 2
    Trainable parameters: (5*5*1+1)*6 = 156
    Connections: 28*28*(5*5+1)*6 = 122304
"""
C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2, bias=True)

"""
Layer S2 (Sub-sampling):
    Input size: 28x28
    Output size: 14x14
    Input channels: 6
    Output channels: 6
    Kernel size: 2x2
    Stride: 2 (non-overlapping)
    Trainable parameters (Original): (1+1)*6 = 12   # Note: MaxPooling/AvgPooling will have 0 parameter
    Connections: 14*14*(2*2+1)*6 = 5880        
"""
S2_maxPooling = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
S2_avgPooling = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

"""
Layer C3 (Convolutional):
    Input size: 14x14
    Output size: 10x10
    Input channels: 6
    Output channels: 16
    Kernel size: 5x5
    Stride: 1
    Trainable parameters (Original): (5*5)*(3*6 + 4*9 + 6*1) + 16 = 1516
    Connections: 10*10*1516 = 151600
"""
C3_simplified = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), bias=True)

"""
Layer S4 (Sub-sampling):
    Input Size: 10x10
    Output Size: 5x5
    Input channels: 16
    Output channels: 16
    Kernel size: 2x2
    Stride: 2 (non-overlapping)
    Trainable parameters(Original): (1+1)*16 = 32
    Connections: 5*5*(2*2+1)*16 = 2000
"""
S4_maxPooling = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
S4_avgPooling = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

"""
Layer C5 (Convolutional): # Fully connected
    Input size: 5x5
    Output size: 5x5
    Input channels: 16*(5*5)
    Output channels: 120
    Kernel size: 1x1
    Stride: 1
    Trainable parameters (Original): (16*5*5+1)*120 = 48120
    Connections: (16*25+1)*120 = 48120  
"""
C5 = nn.Linear(in_features=16 * 5 * 5, out_features=120)

"""
Layer F6 (Fully connected):
    Input size: 5x5
    Output size: 5x5
    Input channels: 120
    Output channels: 84
    Trainable parameters (Original): (120+1)*84 = 10164
    Connections: (120+1)*84 = 10164 
"""
F6 = nn.Linear(in_features=120, out_features=84)

"""
Layer RBF (Radial Basis Function):
    Input size: 5x5
    Output size: 1
    Input channels: 84
    Output channels: 10
"""
RBF = nn.Linear(in_features=84, out_features=10)
