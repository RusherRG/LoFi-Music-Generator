import torch
import random

from dataset import LoFiMIDIDataset
from models.lofi_lstm import LoFiLSTM
from models.constants import *

def lstm_generate(dataset: LoFiMIDIDataset, model_path: str = "./checkpoints/lstm.pth"):
    model = LoFiLSTM(dataset.vocab_size)
    model.load_state_dict(torch.load(model_path))    
    model.eval()

    (melody, _) = dataset[random.randint(0, len(dataset) - 1)]
    generated_melody = melody.tolist()

    for i in range(100):
        melody = melody.reshape(1, MELODY_LENGTH - 1)
        new_note = model(melody)
        new_note = torch.argmax(new_note)
        generated_melody.append(new_note.item())
        melody = melody.tolist()[0]
        melody = melody[1:] + [new_note.item()]
        melody = torch.tensor(melody, dtype=torch.long)

    return generated_melody
