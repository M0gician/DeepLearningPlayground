import torch
import typing
import torch.nn as nn
from torch.optim import SGD
from typing import Union, TypeVar

T = TypeVar('T')
Layer_Type = Union[typing.OrderedDict[str, nn.Module], nn.Module]


class AlexNet(nn.Module):
    def __init__(self, num_classes, verbose=False):
        super(AlexNet, self).__init__()
        self.verbose = verbose
        self.num_classes = num_classes
        self.training = False

        self.convolution = None
        self.classifier = None

        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        """
        The first convolutional layer filters the 224x224x3 input image 
        with 96 kernels of size 11x11x3 with a stride of 4 pixels
        (Note: the actual image size is 227x227x3)
        """
        # Since the 2nd layer has an input of 55x55x48
        # (n_h - k_h + 2*p_h)/s_h + 1 = (227 - 11 + 0)/4 + 1 = 55
        self.C1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4)

        """
        The second convolutional layer takes as input the (response-normalized and pooled) 
        output of the first layer and filters it with 256 kernels of size 5x5x48.
        """
        # We used k = 2, n = 5, alpha = 10^-4, and beta = 0.75.
        # We applied this normalization after applying the ReLU non-linearity in certain layers
        self.RN2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        # (n_h - k_h + 2*p_h)/s_h + 1 = (55 - 3 + 0)/2 + 1 = 27
        self.P2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        # (n_h - k_h + 2*p_h)/s_h + 1  = (27 - 5 + 2*2)/1 + 1 = 27
        self.C2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=(2, 2))

        """
        The third convolutional layer has 384 kernels of size 3x3x256 connected to 
        the (normalized, pooled) outputs of the second convolutional layer. 
        """
        self.RN3 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        # (n_h - k_h + 2*p_h)/s_h + 1 = (27 - 3 + 0)/2 + 1 = 13
        self.P3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        # (n_h - k_h + 2*p_h)/s_h + 1 = (13 - 3 + 2*1)/1 + 1 = 13
        self.C3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=(1, 1))

        """
        The fourth convolutional layer has 384 kernels of size 3x3x192
        """
        # (n_h - k_h + 2*p_h)/s_h + 1 = (13 - 3 + 2*1)/1 + 1 = 13
        self.C4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=(1, 1))

        """
        The fifth convolutional layer has 256 kernels of size 3x3x192
        """
        # (n_h - k_h + 2*p_h)/s_h + 1 = (13 - 3 + 2*1)/1 + 1 = 13
        self.C5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        # (n_h - k_h + 2*p_h)/s_h + 1 = (13 - 3 + 0)/2 + 1 = 6
        self.P5 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        """
        The fully-connected layers have 4096 neurons each
        """
        self.F6 = nn.Linear(in_features=(256 * 6 * 6), out_features=4096)
        self.F7 = nn.Linear(in_features=4096, out_features=4096)
        self.F8 = nn.Linear(in_features=4096, out_features=self.num_classes)

        self.convolution = nn.Sequential(
            self.C1,
            self.C2, nn.ReLU(inplace=True), self.RN2, self.P2,
            self.C3, nn.ReLU(inplace=True), self.RN3, self.P3,
            self.C4,
            self.C5, self.P5
        )

        """
        The ReLU non-linearity is applied to the output of every convolutional
            and fully-connected layer.
        Dropout is used in the first two fully-connected layers, consisting of 
            setting to zero the output of each hidden neuron with probability 0.5.
        """
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            self.F6,
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            self.F7,
            nn.ReLU(inplace=True),
            self.F8
        )

    def initialize(self, criterion=None, optimizer=None, scheduler=None, learning_rate=0.01) -> None:
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion  = criterion

        """
        We trained our models using stochastic gradient descent with a batch size of 128 examples,
            momentum of 0.9, and weight decay of 0.0005.
        We used an equal learning rate for all layers, which we adjusted manually throughout training.
        The heuristic which we followed was to divide the learning rate by 10 when the validation 
            error rate stopped improving with the current learning rate. 
        The learning rate was initialized at 0.01 and reduced three times prior to termination.
        """
        if optimizer is None:
            self.optimizer = SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
        else:
            self.optimizer = optimizer

        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='min', factor=0.1, patience=5, threshold=0.002
            )
        else:
            self.scheduler = scheduler

        """
        We initialized the weights in each layer from a zero-mean Gaussian distribution 
            with standard deviation 0.01.
        We initialized the neuron biases in the second, fourth, and fifth convolutional layers, 
            as well as in the fully-connected hidden layers, with the constant 1.
        We initialized the neuron biases in the remaining layers with the constant 0.
        """
        for name, module in self.convolution.named_children():
            if type(module) == nn.Conv2d:
                nn.init.normal_(tensor=module.weight, mean=0, std=0.01)
                if name in ['0', '2']:
                    nn.init.constant_(tensor=module.bias, val=0)
                else:
                    nn.init.constant_(tensor=module.bias, val=1)

        for name, module in self.classifier.named_children():
            if type(module) == nn.Linear:
                nn.init.normal_(tensor=module.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=module.bias, val=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.convolution(X)
        X = X.view(-1, 256*6*6)
        return self.classifier(X)

    def train(self, mode=True, data=None, epochs=10) -> 'AlexNet':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

        if data is None:
            raise FileNotFoundError("\"data\" has to be a valid Dataloader object!")

        self.training = mode
        for module in self.convolution:
            module.train(mode)
        for module in self.classifier:
            module.train(mode)

        running_loss = 0.0
        for epoch in range(0, epochs):
            for i, datum in enumerate(data, 0):
                features, labels = datum[0].to(device), datum[1].to(device)
                loss = self.criterion(self(features), labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                batch_split = int(len(data.dataset) / data.batch_size / 5)
                batch_split = 1 if batch_split < 1 else batch_split
                if i % batch_split == batch_split - 1:
                    if self.verbose:
                        print(f"[epoch {epoch + 1}, batch {i + 1}] loss: {running_loss / batch_split}")
                    self.scheduler.step(running_loss / batch_split)
                    running_loss = 0.0

        if self.verbose:
            print('Finished Training')
        return self
