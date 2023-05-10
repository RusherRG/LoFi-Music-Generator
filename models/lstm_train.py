import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LoFiMIDIDataset
from models.lofi_lstm import LoFiLSTM
from models.constants import *


def lstm_train(
    model: LoFiLSTM,
    data_loader: DataLoader,
    device: str,
    epochs: int = 100,
):
    train_losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    print(f"Training model: {type(model).__name__}")
    print(f"Criterion: {type(criterion).__name__}")
    print(f"Optimizer: {type(optimizer).__name__}")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        model.train()
        epoch_loss = 0
        for inputs, target in data_loader:
            inputs, target = inputs.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Training loss: {epoch_loss:.4f}")
        train_losses.append(epoch_loss)

    torch.save(model.state_dict(), "./checkpoints/lstm.pth")
