from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator


if __name__ == '__main__':
    args = TrainOptions().parse()
    epochs = args.epochs
    batch_size = args.batch_size
    
