import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LoFiMIDIDataset
from models.lofi_lstm import LoFiLSTM
from models.lofi_vae import LoFiVAE
from models.lstm_train import lstm_train
from models.vae_train import vae_train
from models.constants import *


def train(model_name: str, dataset_dir: str = "./dataset", epochs: int = 100):
    """
    Runs training based on the `model_name` on the dataset present in the `dataset_dir`
    Supported models should be added in ./models/
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = LoFiMIDIDataset(dataset_dir=dataset_dir)
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Dataset size: {len(dataset)}")

    # create data loader
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    # initialize the model object
    if model_name == "lstm":
        model = LoFiLSTM(dataset.vocab_size)
    elif model_name == "vae":
        model = LoFiVAE(dataset.vocab_size, device)
    print(model)
    # run the trainer of the model
    model = model.to(device)
    if model_name == "lstm":
        lstm_train(model, data_loader, device, epochs)
    elif model_name == "vae":
        vae_train(model, data_loader, device, epochs)


if __name__ == "__main__":
    train(model_name="vae", dataset_dir="./dataset", epochs=500)
