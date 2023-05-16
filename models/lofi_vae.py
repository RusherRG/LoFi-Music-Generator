from hashlib import md5
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .constants import *


class LoFiVAE(nn.Module):
    def __init__(self, vocab_size, device):
        super(LoFiVAE, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.encoder = Encoder(vocab_size, device)
        self.decoder = Decoder(vocab_size, device)
        self.mean_linear = nn.Linear(in_features=SAMPLE_SIZE, out_features=SAMPLE_SIZE)
        self.variance_linear = nn.Linear(
            in_features=SAMPLE_SIZE, out_features=SAMPLE_SIZE
        )

    def forward(
        self,
        chords,
        offsets,
        durations,
        num_chords,
        sampling_rate_chords=0,
    ):
        # encode
        h = self.encoder(chords, offsets, durations)
        # VAE
        mu = self.mean_linear(h)
        log_var = self.variance_linear(h)
        z = self.sample(mu, log_var)
        # compute the Kullback–Leibler divergence between a Gaussian and an uniform Gaussian
        kl = 0.5 * torch.mean(mu**2 + log_var.exp() - log_var - 1, dim=[0, 1])
        # decode
        if self.training:
            (
                chord_outputs,
                offset_outputs,
                duration_outputs,
            ) = self.decoder(z, sampling_rate_chords, chords)
        else:
            (
                chord_outputs,
                offset_outputs,
                duration_outputs,
            ) = self.decoder(z)

        return chord_outputs, offset_outputs, duration_outputs, kl

    # reparameterization trick:
    # because backpropagation cannot flow through a random node, we introduce a new parameter that allows us to
    # reparameterize z in a way that allows backprop to flow through the deterministic nodes
    # https://stats.stackexchange.com/questions/199605/how-does-the-reparameterization-trick-for-vaes-work-and-why-is-it-important
    def sample(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * (logvar / 2).exp()
        else:
            return mu


class Encoder(nn.Module):
    def __init__(self, vocab_size, device):
        super(Encoder, self).__init__()
        self.device = device
        self.vocab_size = vocab_size

        # we need chord embedding to convert the string notes to an embedding
        self.chord_embeddings = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=CHORD_EMBEDDING_SIZE
        )
        self.chord_lstm = nn.LSTM(
            input_size=CHORD_EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            bidirectional=True,
            batch_first=True,
        )
        self.offset_lstm = nn.LSTM(
            input_size=CHORD_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            bidirectional=True,
            batch_first=True,
        )
        self.duration_lstm = nn.LSTM(
            input_size=CHORD_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            bidirectional=True,
            batch_first=True,
        )

        self.linear = nn.Linear(in_features=4 * HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        self.downsample = nn.Linear(in_features=HIDDEN_SIZE, out_features=SAMPLE_SIZE)

    def forward(self, chords, offsets, durations):
        chord_embeddings = self.chord_embeddings(chords)
        chords_out, (h_chords, _) = self.chord_lstm(chord_embeddings)
        offsets_out, (h_offsets, _) = self.offset_lstm(offsets.unsqueeze(1))
        durations_out, (h_durations, _) = self.duration_lstm(durations.unsqueeze(1))

        h_concatenated = torch.cat((h_chords[-1], h_chords[-2]), dim=1)
        linear_out = self.linear(
            torch.cat(
                (h_concatenated, h_offsets[-1], h_durations[-1]), dim=1
            )
        )
        sample = self.downsample(linear_out)
        return sample


class Decoder(nn.Module):
    def __init__(self, vocab_size, device):
        super(Decoder, self).__init__()
        self.device = device
        self.vocab_size = vocab_size

        # we need chord embedding to convert the string notes to an embedding
        self.chord_lstm = nn.LSTMCell(input_size=SAMPLE_SIZE, hidden_size=HIDDEN_SIZE)
        self.chord_embeddings = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=CHORD_EMBEDDING_SIZE
        )
        self.chord_prediction = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE, out_features=self.vocab_size),
        )
        self.chord_embedding_downsample = nn.Linear(
            in_features=SAMPLE_SIZE + CHORD_EMBEDDING_SIZE, out_features=SAMPLE_SIZE
        )

        self.offset_lstm = nn.LSTMCell(
            input_size=CHORD_EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE
        )
        self.offset_prediction = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE2, out_features=1),
        )

        self.duration_lstm = nn.LSTMCell(
            input_size=CHORD_EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE
        )
        self.duration_prediction = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE2, out_features=1),
        )

    def decode(self, mu, num_chords=CHORD_LENGTH):
        # create a hash for vector mu
        hash = ""
        # first 20 characters are each sampled from 5 entries
        for i in range(0, 100, 5):
            hash += str((mu[0][i : i + 1].abs().sum() * 587).int().item())[-1]
        # last 4 characters are the beginning of the MD5 hash of the whole vector
        hash2 = int(md5(mu.numpy()).hexdigest(), 16)
        hash = f"#{hash}{hash2}"[:25]
        mu = mu.to(self.device)
        return hash, self(mu, num_chords=num_chords)

    def forward(
        self,
        z,
        sampling_rate_chords=0,
        gt_chords=None,
        gt_offsets=None,
        gt_durations=None,
        num_chords=CHORD_LENGTH,
    ):
        batch_size = z.shape[0]
        # initialize hidden states and cell states
        hx_chords = torch.zeros(batch_size, HIDDEN_SIZE, device=self.device)
        cx_chords = torch.zeros(batch_size, HIDDEN_SIZE, device=self.device)

        hx_offsets = torch.zeros(batch_size, HIDDEN_SIZE, device=self.device)
        cx_offsets = torch.zeros(batch_size, HIDDEN_SIZE, device=self.device)

        hx_durations = torch.zeros(batch_size, HIDDEN_SIZE, device=self.device)
        cx_durations = torch.zeros(batch_size, HIDDEN_SIZE, device=self.device)

        chord_outputs = []
        offset_outputs = []
        duration_outputs = []

        # the chord LSTM input at first only consists of z
        # after the first iteration, we use the chord embeddings
        chord_embeddings = z

        for i in range(num_chords):
            hx_chords, cx_chords = self.chord_lstm(
                chord_embeddings, (hx_chords, cx_chords)
            )
            chord_prediction = self.chord_prediction(hx_chords)
            chord_outputs.append(chord_prediction)

            # perform teacher forcing during training
            perform_teacher_forcing_chords = bool(
                np.random.choice(
                    2, 1, p=[1 - sampling_rate_chords, sampling_rate_chords]
                )[0]
            )
            if gt_chords is not None and perform_teacher_forcing_chords:
                chord_embeddings = self.chord_embeddings(gt_chords[:, i])
            else:
                chord_embeddings = self.chord_embeddings(chord_prediction.argmax(dim=1))

            hx_offsets, cx_offsets = self.offset_lstm(chord_embeddings)
            offsets_prediction = self.offset_prediction(hx_offsets)
            offset_outputs.append(offsets_prediction)

            hx_durations, cx_durations = self.duration_lstm(chord_embeddings)
            durations_prediction = self.duration_prediction(hx_durations)
            duration_outputs.append(durations_prediction)

            # # let z influence the chord embedding
            chord_embeddings = self.chord_embedding_downsample(
                torch.cat((chord_embeddings, z), dim=1)
            )

        chord_outputs = torch.stack(chord_outputs, dim=2)
        offset_outputs = torch.stack(offset_outputs, dim=2)
        duration_outputs = torch.stack(duration_outputs, dim=2)

        return chord_outputs, offset_outputs, duration_outputs
