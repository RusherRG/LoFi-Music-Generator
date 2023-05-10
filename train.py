import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LoFiMIDIDataset
from models.lstm import LoFiLSTM
from models.lstm_train import lstm_train
from model.constants import *


def train(model_name: str, dataset_dir: str = "./dataset"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = LoFiMIDIDataset(dataset_dir=dataset_dir)
    train_data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    if model_name == "lstm":
        model = LoFiLSTM(dataset.vocab_size)

    model = model.to(device)
    if model_name == "lstm":
        lstm_train(dataset, model)


if __name__ == "__main__":
    train("lstm")
