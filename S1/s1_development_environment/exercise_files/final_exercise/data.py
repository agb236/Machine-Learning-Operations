import torch
import matplotlib.pyplot as plt
import random

def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    data_path = r"C:\Users\emilg\Desktop\DTU\1. Semester\Machine Learning Operations\corruptmnist_v1-20250106T101947Z-001\corruptmnist_v1"
    train_images = []
    train_targets = []
    
    for i in range(5):  # Loop over the 5 files
        train_images.append(torch.load(f"{data_path}\\train_images_{i}.pt"))
        train_targets.append(torch.load(f"{data_path}\\train_target_{i}.pt"))
    
    # Concatenate the loaded tensors
    train_images = torch.cat(train_images, dim=0)
    train_targets = torch.cat(train_targets, dim=0)

    # Load test data
    test_images = torch.load(f"{data_path}\\test_images.pt")
    test_targets = torch.load(f"{data_path}\\test_target.pt")
    
    return [train_images, train_targets], [test_images, test_targets]

train, test = corrupt_mnist()
train_images = train[0]
train_targets = train[1]
# Plot one of the images
def plot_image(images, index):
    """Plot a single MNIST image."""
    image = images[index]  # Select the image
    plt.imshow(image.squeeze().numpy(), cmap="gray")  # Remove channel dimension and plot
    plt.title(f"Label: {train_targets[index].item()}")
    plt.axis("off")
    plt.show()

# Plot the first image
plot_image(train_images, index=random.randint(0, 25000))