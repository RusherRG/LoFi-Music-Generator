import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LoFiMIDIDataset
from models.lstm import LoFiLSTM
from models.constants import *


def lstm_train(
    dataset: LoFiMIDIDataset,
    model: LoFiLSTM,
):
    train_losses = []
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    print(f"Training model: {type(model).__name__}")
    print(f"Loss: {type(loss).__name__}")
    print(f"Optimizer: {type(optimizer).__name__}")

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        model.train()
        epoch_loss = 0
        for inputs, target in train_data_loader:
            outputs = model(inputs)

            optimzer.zero_grad()
            loss.backward()

            optimzer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss)
