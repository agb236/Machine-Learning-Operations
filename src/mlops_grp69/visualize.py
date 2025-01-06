import os
import torch
import torchvision
from torchvision import models, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from data import corrupt_mnist
from model import MyAwesomeModel
import os
from datetime import datetime


# Define paths
PROJECT_NAME = "mlops_grp69" 
REPORTS_DIR = os.path.join("reports", "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Parameters
BATCH_SIZE = 32
TSNE_PERPLEXITY = 30
TSNE_ITER = 1000

# Load the pre-trained network
def load_model():
    model = models.resnet18(pretrained=True)
    # Remove the final classification layer (fully connected)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model