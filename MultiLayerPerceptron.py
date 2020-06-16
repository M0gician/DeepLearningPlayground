import torch
import typing
import operator
import torch.nn as nn
from torch.optim import SGD
from itertools import islice
from collections import OrderedDict
from torch.utils.data import DataLoader
from typing import Union, TypeVar, Iterator

T = TypeVar('T')
Layer_Type = Union[typing.OrderedDict[str, nn.Module], nn.Module]


class FunctionModifiers(object):
    """
    Used to denote the behavior of a function in TorchScript. See export() and
    ignore() for details.
    """
    UNUSED = "unused (ignored and replaced with raising of an exception)"
    IGNORE = "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
    EXPORT = "export (compile this function even if nothing calls it)"
    DEFAULT = "default (compile if called from a exported function / forward)"
    COPY_TO_SCRIPT_WRAPPER = \
        "if this method is not scripted, copy the python method onto the scripted model"


def _copy_to_script_wrapper(fn):
    fn._torchscript_modifier = FunctionModifiers.COPY_TO_SCRIPT_WRAPPER
    return fn


class Sequential(nn.Module):
    def __init__(self, *layers: Layer_Type):
        self.training = False

        self.criterion = None
        self.optimizer = None

        super(Sequential, self).__init__()
        if len(layers) == 1 and isinstance(layers[0], OrderedDict):
            for name, module in layers[0].items():
                self.add_module(name, module)
        else:
            for idx, module in enumerate(layers):
                self.add_module(str(idx), module)
    
    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)
    
    @_copy_to_script_wrapper
    def __dir__(self) -> list:
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._modules.values())

    def _get_item_by_idx(self, iterator, idx):
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f"index {idx} is out of range")
        idx %= size
        return next(islice(iterator, idx, None))   

    @_copy_to_script_wrapper
    def __getitem__(self: T, idx: Union[slice, int]) -> T:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items()))[idx])
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: nn.Module) -> None:
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def initialize(self, criterion=None, optimizer=None, learning_rate=0.5) -> None:
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        if optimizer is None:
            self.optimizer = SGD(self.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for name, module in self._modules.items():
            if name != "criterion":
                X = module(X)
        return X
    
    def train(self, mode=True) -> None:
        self.training = mode
        for module in self:
            module.train(mode)


class MLP(Sequential):
    def __init__(self, *layers: Layer_Type, verbose=False):
        super(MLP, self).__init__()
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

    def train(self, mode=True, data=None, epochs=10) -> 'MLP':
        if data is None:
            raise FileNotFoundError("\"data\" has to be a valid Dataloader object!")
        if self.verbose:
            running_loss = 0.0

        self.training = mode
        for module in self:
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
                    if i % batch_split == batch_split - 1:
                        print(f"[epoch {epoch + 1}, batch {i + 1}] loss: {running_loss / batch_split}")
                        running_loss = 0.0

        if self.verbose:
            print('Finished Training')
        return self


if __name__ == "__main__":
    from helper import load_fashion_mnist_data, predict_fashion_mnist
    from helper import evaluate_accuracy, evaluate_loss

    # Input size: 28 (width) * 28 (height) = 784 pixels
    num_inputs = 784

    # Output size: 10 categories
    num_outputs = 10

    batch_size = 256
    mnist_train, mnist_test = load_fashion_mnist_data(batch_size, resize=None)

    class Reshape(nn.Module):
        def __init__(self, in_features: int):
            super(Reshape, self).__init__()
            self.in_features = in_features

        def forward(self, X):
            return X.view(-1, self.in_features)
        
    rsp = Reshape(num_inputs)
    fc1 = nn.Linear(num_inputs, 256)
    relu = nn.ReLU()
    fc2 = nn.Linear(256, num_outputs)

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.zeros_(m.bias)

    mlpNet = MLP(rsp, fc1, relu, fc2, verbose=True)
    mlpNet.initialize()
    mlpNet.apply(init_weights)
    mlpNet.train(mode=True, data=mnist_train, epochs=5)

