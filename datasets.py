import os

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision import transforms
import tqdm

import pytorch_lightning as pl

def train_dataloader(batch_size):
    # REQUIRED
    #dataset = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
    #dataset = CIFAR10(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
    #loader = DataLoader(dataset, batch_size=32)

    # Downloading/Louding CIFAR10 data
    trainset  = CIFAR10(root=os.getcwd(), train=True , download=False)#, transform = transform_with_aug)
    testset   = CIFAR10(root=os.getcwd(), train=False, download=False)#, transform = transform_no_aug)
    classDict = {'plane':0, 'car':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

    # Separating trainset/testset data/label
    x_train  = trainset.data
    x_test   = testset.data
    y_train  = trainset.targets
    y_test   = testset.targets

    # Define a function to separate CIFAR classes by class index

    def get_class_i(x, y, i):
        """
        x: trainset.train_data or testset.test_data
        y: trainset.train_labels or testset.test_labels
        i: class label, a number between 0 to 9
        return: x_i
        """
        # Convert to a numpy array
        y = np.array(y)
        # Locate position of labels that equal to i
        pos_i = np.argwhere(y == i)
        # Convert the result into a 1-D list
        pos_i = list(pos_i[:,0])
        # Collect all data that match the desired label
        x_i = [x[j] for j in pos_i]
        
        return x_i

    class DatasetMaker(Dataset):
        def __init__(self, datasets, transformFunc = transforms.ToTensor()):
            """
            datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
            """
            self.datasets = datasets
            self.lengths  = [len(d) for d in self.datasets]
            self.transformFunc = transformFunc
        def __getitem__(self, i):
            class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
            img = self.datasets[class_label][index_wrt_class]
            if self.transformFunc:
              img = self.transformFunc(img)
            return img, class_label

        def __len__(self):
            return sum(self.lengths)
        
        def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
            """
            Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
            """
            # Which class/bin does i fall into?
            accum = np.add.accumulate(bin_sizes)
            if verbose:
                print("accum =", accum)
            bin_index  = len(np.argwhere(accum <= absolute_index))
            if verbose:
                print("class_label =", bin_index)
            # Which element of the fallent class/bin does i correspond to?
            index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
            if verbose:
                print("index_wrt_class =", index_wrt_class)

            return bin_index, index_wrt_class

    # ================== Usage ================== #

    # Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
    cat_dog_trainset = \
        DatasetMaker([get_class_i(x_train, y_train, classDict['cat'])
              , get_class_i(x_train, y_train, classDict['dog'])]
            #transform_with_aug
        )
    cat_dog_testset  = \
        DatasetMaker(
            [get_class_i(x_test , y_test , classDict['cat']), get_class_i(x_test , y_test , classDict['dog'])]
            #transform_no_aug
        )

    kwargs = {'num_workers': 2, 'pin_memory': False}

    # Create datasetLoaders from trainset and testset
    trainsetLoader   = DataLoader(cat_dog_trainset, batch_size=batch_size, shuffle=True , **kwargs)
    testsetLoader    = DataLoader(cat_dog_testset , batch_size=batch_size, shuffle=False, **kwargs)

    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # self.data_np = np.array([img.numpy() for img, label in dataset])
    #self._train_loader = loader
    #return loader
    return trainsetLoader

def get_samples(count, only_labels=None, resolution=32, channels=3):
  loader = train_dataloader(1)
  it = iter(loader)
  results_images = []
  results_labels = []
  for i in range(count):
    images, labels = next(it)
    results_images.append(images.numpy())
    results_labels.append(labels.numpy())
  images = np.concatenate(results_images, axis=0)
  labels = np.concatenate(results_labels, axis=0)
  return images, labels

