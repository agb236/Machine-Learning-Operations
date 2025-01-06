import torch
import typer
from torch import nn
from torch import optim
from data_solution import corrupt_mnist
from model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
app = typer.Typer()


@app.command()
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
            # Flatten MNIST images into a 784-long vector
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch: {epoch}/{epochs} - Loss: {running_loss/epoch}")



@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(f"Loading model from checkpoint: {model_checkpoint}")

    # Load the trained model
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()  # Set the model to evaluation mode

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    # Flatten the test images into 784-long vectors
    test_images = test_images.view(test_images.shape[0], -1)

    # Disable gradient calculations for evaluation
    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")



if __name__ == "__main__":
    app() 
