import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .constants import *

class LoFiLSTM(nn.Module):
    def __init__(self, vocab_size: int):
        super(LoFiLSTM, self).__init__()
        self.vocab_size = vocab_size

        self.melody_embedding = nn.Embedding(num_embeddings=MELODY_LENGTH, embedding_dim=MELODY_EMBEDDING_SIZE)
        self.melody_lstm = nn.LSTM(input_size=MELODY_EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE_1, num_layers=NUM_LAYERS, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(in_features=HIDDEN_SIZE_1, out_features=HIDDEN_SIZE_2)
        self.output = nn.Linear(in_features=HIDDEN_SIZE_2, out_features=self.vocab_size)

    def forward(self, melody):
        embedding_out = self.melody_embedding(melody)
        lstm_out = self.melody_lstm(embedding_out)
        linear_out = self.linear(lstm_out)
        output = self.output(linear_out)
        return output
:
    def __init__(self):
        self.lstm = nn.LSTM()