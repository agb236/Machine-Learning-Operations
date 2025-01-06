import torch
import matplotlib.pyplot as plt
import random

def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    Adam = 0
    if Adam == 1:
        data_path = r"/workspaces/Machine-Learning-Operations/data/corruptmnist/"
    else: data_path = r"data/corruptmnist/"
    train_images = []
    train_targets = []
    
    for i in range(6):  # Loop over the 5 files
        train_images.append(torch.load(f"{data_path}train_images_{i}.pt"))
        train_targets.append(torch.load(f"{data_path}train_target_{i}.pt"))
    
    # Concatenate the loaded tensors
    train_images = torch.cat(train_images, dim=0)
    train_targets = torch.cat(train_targets, dim=0)

    # Load test data
    test_images = torch.load(f"{data_path}test_images.pt")
    test_targets = torch.load(f"{data_path}test_target.pt")
    
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

if __name__ == "__main__":
    # Plot the first image
    plot_image(train_images, index=random.randint(0, 25000))