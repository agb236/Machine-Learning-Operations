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
    model = MyAwesomeModel()
    train_set, _ = corrupt_mnist()
    # Remove the final classification layer (fully connected)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

def extract_features(model, dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in dataloader:
            outputs = model(images).squeeze()
            features.append(outputs.cpu().numpy())
            labels.append(lbls.cpu().numpy())
    return np.vstack(features), np.hstack(labels)

def visualize_tsne(features, labels, output_file):
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, n_iter=TSNE_ITER, random_state=42)
    reduced_features = tsne.fit_transform(features)

    print("Plotting t-SNE visualization...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=5, alpha=0.8)
    plt.colorbar(scatter, label='Class Label')
    plt.title("t-SNE Visualization of CNN Features")
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")
    plt.close()
    
def main():
    # Define the data transformation and dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Example: Using CIFAR-10 as training data
    dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model and extract features
    model = load_model()
    features, labels = extract_features(model, dataloader)

    # Save visualization
    output_file = os.path.join(REPORTS_DIR, "tsne_visualization.png")
    visualize_tsne(features, labels, output_file)

if __name__ == "__main__":
    main()