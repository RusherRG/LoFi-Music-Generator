import pickle
import torch

from torch import nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LoFiMIDIDataset
from models.lofi_vae import LoFiVAE
from models.constants import *


def vae_train(
    model: LoFiVAE,
    train_data_loader: DataLoader,
    device: str,
    epochs: int = 100,
):
    criterion = nn.CrossEntropyLoss(reduction="mean")
    l1_loss = nn.L1Loss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_losses = []
    val_losses = []
    sampling_rate_chords = 0

    print(f"Training model: {type(model).__name__}")
    print(f"Criterion: {type(criterion).__name__}")
    print(f"Optimizer: {type(optimizer).__name__}")

    def compute_loss(notes, offsets, durations):
        notes = notes.to(device)
        offsets = offsets.to(device)
        durations = durations.to(device)

        pred_notes, pred_offsets, pred_durations, kl = model(
            notes, offsets, durations, sampling_rate_chords
        )
        loss_kl = kl
        loss_notes = criterion(pred_notes, notes)
        loss_offsets = l1_loss(pred_offsets, offsets)
        loss_durations = l1_loss(pred_durations, durations)

        loss_total = loss_notes + loss_offsets + loss_durations + loss_kl

        return loss_total, loss_notes, loss_offsets, loss_durations, loss_kl

    for epoch in range(epochs):
        print(f"\n\nEpoch {epoch+1}")
        (
            ep_train_losses,
            ep_train_losses_notes,
            ep_train_losses_offsets,
            ep_train_losses_durations,
            ep_train_losses_kl,
        ) = ([], [], [], [], [])
        if TEACHER_FORCE:
            sampling_rate_chords = sampling_rate_at_epoch(epoch)

        print(f"Scheduled sampling rate: {sampling_rate_chords}")

        model.train()
        print("Training")
        for batch, (notes, offsets, durations) in enumerate(train_data_loader):
            (
                loss,
                loss_notes,
                loss_offsets,
                loss_durations,
                loss_kl,
            ) = compute_loss(notes, offsets, durations)

            ep_train_losses.append(loss.item())
            ep_train_losses_notes.append(loss_notes.item())
            ep_train_losses_offsets.append(loss_offsets.item())
            ep_train_losses_durations.append(loss_durations.item())
            ep_train_losses_kl.append(loss_kl.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                f"Batch {batch}:\tTraining loss: {loss:.3f} (N: {loss_notes:.3f} + O: {loss_offsets:.3f} + D: {loss_durations:.3f} + KL: {loss_kl:.3f})"
            )
        train_losses.append((ep_train_losses, ep_train_losses_notes, ep_train_losses_offsets, ep_train_losses_durations, ep_train_losses_kl))

    torch.save(model.state_dict(), "./checkpoints/vae.pth")

    with open("./checkpoints/losses", "wb") as f:
        pickle.dump(train_losses, f)
