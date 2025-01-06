import torch
import typer
from torch import nn
from torch import optim
from data import corrupt_mnist
from model import MyAwesomeModel
import os
from datetime import datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr: float = 1e-3) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = corrupt_mnist()
    # Define hyper parameters
    epochs = 10
    learning_rate = 0.003
    batch_size = 32
    # Define the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_dataloader:
            # Remove the flattening of MNIST images
            # images = images.view(images.shape[0], -1)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch: {epoch+1}/{epochs} - Loss: {running_loss/epochs}")

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)
    # Generate a new filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"models/model_{timestamp}.pth"

    # Save the trained model
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    typer.run(train)
