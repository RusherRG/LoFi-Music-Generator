import torch
import random

from dataset import LoFiMIDIDataset
from models.lofi_vae import LoFiVAE
from models.constants import *


def vae_generate(dataset: LoFiMIDIDataset, model_path: str = "./checkpoints/vae.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model")
    model = LoFiVAE(dataset.vocab_size, device)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    print("Predicting notes")
    number_list = [random.uniform(-2, 2) for i in range(SAMPLE_SIZE)]
    mu = torch.tensor([number_list]).float()
    hash_, (pred_notes, pred_offsets, pred_durations) = model.decoder.decode(mu, 100)
    notes = pred_notes.argmax(dim=1)[0].tolist()
    offsets = pred_offsets[0][0].tolist()
    durations = pred_durations[0][0].tolist()
    
    return notes, offsets, durations
