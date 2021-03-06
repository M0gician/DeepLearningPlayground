import torch
import torchvision
import os
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt

FASHION_MINST_LABELS = [
    't-shirt', 'trouser', 'pullover', 'dress', 'coat',
    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Accumulator: 
    """Sum a list of numbers over time."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    y_hat = y_hat.cpu()
    if y_hat.shape[1] > 1:
        return float((y_hat.argmax(axis=1).type(torch.float32) ==
                      y.type(torch.float32)).sum())
    else:
        return float((y_hat.type(torch.int32) == y.type(torch.int32)).sum())


def evaluate_accuracy(net, data_iter):  
    metric = Accumulator(2)  # num_corrected_examples, num_examples
    for X, y in data_iter:
        X_gpu = X.to(device)
        # X, y = X.to(device), y.to(device)
        metric.add(accuracy(net(X_gpu), y), y.numpy().size)
    return metric[0] / metric[1]


def evaluate_loss(net, data_iter):  #@save
    """Evaluate the loss of a model on the given dataset."""
    metric = Accumulator(2)  # sum_loss, num_examples
    for X, y in data_iter:
        X_gpu, y_gpu = X.to(device), y.to(device)
        # X, y = X.to(device), y.to(device)
        l_gpu = net.criterion(net(X_gpu), y_gpu)
        l = l_gpu.cpu()
        if l.nelement() != 1:
            metric.add(l.sum(), y.numpy().size)
        else:
            metric.add(l*len(y), y.numpy().size)
    return metric[0] / metric[1]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):        
        if 'asnumpy' in dir(img): img = img.asnumpy() 
        if 'numpy' in dir(img): img = img.cpu().numpy()
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def synthetic_data(w, b, num_examples):
    X = torch.zeros(size=(num_examples, w.shape[0])).normal_()
    y = X @ w + b
    y += torch.zeros(size=y.shape).normal_(std=0.01)
    return X, y


def load_tiny_imagenet(batch_size: int, IMAGENET_DIR='tiny-imagenet-200'):
    dirname = os.path.dirname(__file__)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale(size=(227, 227)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Scale(size=(227,227)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Scale(size=(227, 227)),
            transforms.ToTensor(),
        ])
    }

    num_workers = {
        'train': 8,
        'val': 8,
        'test': 8
    }

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(dirname, IMAGENET_DIR, x),
            data_transforms[x]
        ) for x in ['train', 'val', 'test']
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers[x]
        ) for x in ['train', 'val', 'test']
    }

    return dataloaders


def load_fashion_mnist_data(batch_size, resize=None):
    if resize is None:
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        trans = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])

    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    
    return DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),\
           DataLoader(mnist_test, batch_size, shuffle=True, num_workers=4)


def predict_fashion_mnist(net: torch.nn.Module, dataset: DataLoader, n=6):
    for features, labels in dataset:
        break
    trues = [FASHION_MINST_LABELS[int(i)] for i in labels]
    features_gpu = features.to(device)
    raw_preds = net(features_gpu).argmax(axis=1)
    preds = [FASHION_MINST_LABELS[int(i)] for i in raw_preds]
    titles = [true+'\n' + pred for true, pred in zip(trues, preds)]
    show_images(features[0:n].reshape(n, 28, 28), 1, n, titles=titles[0:n])