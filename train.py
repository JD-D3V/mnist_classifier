import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from data_loader import get_dataloaders


def train_model(epochs=5, lr=0.001):
    print("Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    train_loader, _ = get_dataloaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "mnist_cnn.pth")


if __name__ == "__main__":
    train_model()
