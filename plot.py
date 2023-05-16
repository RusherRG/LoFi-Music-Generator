import random
import pickle
import matplotlib.pyplot as plt

# Generate some sample data
epoch = [i + 1 for i in range(500)]
loss = []
loss_notes = []
loss_offsets = []
loss_durations = []
loss_kl = []
with open("./checkpoints/losses", "rb") as f:
    train_losses = pickle.load(f)
    for i, (l, ln, lo, ld, ll) in enumerate(train_losses):
        loss_notes.append(sum(ln) / len(ln))
        loss_offsets.append(sum(lo) / len(lo))
        loss_durations.append(sum(ld) / len(ld))
        loss_kl.append(sum(ll) / len(ll))
        loss.append(loss_notes[-1] + loss_durations[-1] + loss_offsets[-1] + loss_kl[-1])

# Create the first subplot (4 lines on one graph)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epoch, loss_notes, label="Notes Loss")
ax.plot(epoch, loss_offsets, label="Offsets Loss")
ax.plot(epoch, loss_durations, label="Durations Loss")
ax.plot(epoch, loss_kl, label="KL Divergence")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Losses vs Epochs")
ax.legend()
fig.savefig("loss.png")

# Create the second subplot (1 line)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epoch, loss)
ax.set_xlabel("Epochs")
ax.set_ylabel("Total Loss")
ax.set_title("Total Loss vs Epochs")

# Show the plots
fig.savefig("total_loss.png")