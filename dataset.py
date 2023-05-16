import os
import pickle

import music21
import torch

from torch.utils.data import Dataset
from models.constants import *

from fractions import Fraction


class LoFiMIDIDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.data, self.vocab = self._load_data()
        self.vocab_size = len(self.vocab)
        self.dataset = self._preprocess(self.data)

    def _convert_to_float(self, fraction):
        """
        Convert fraction to float
        """
        if isinstance(fraction, Fraction):
            return fraction.numerator / fraction.denominator
        return fraction

    def _process_midi(self, midi: music21.stream.Score):
        """
        Process a MIDI file to extract the required data features like
        - note
        - duration
        - volume
        - octave
        """
        notesAndRests = midi.flat.notesAndRests
        notes = []
        offsets = []
        durations = []

        prev_offset = None
        for part in notesAndRests:
            if prev_offset is None:
                prev_offset = self._convert_to_float(part.offset)
            current_offset = self._convert_to_float(part.offset)
            offset = (current_offset - prev_offset)*5
            if isinstance(part, music21.note.Note):
                notes.append(part.pitch.nameWithOctave)
                offsets.append(offset)
                durations.append(self._convert_to_float(part.duration.quarterLength)*5)
            elif isinstance(part, music21.note.Rest):
                notes.append("R")
                offsets.append(offset)
                durations.append(self._convert_to_float(part.duration.quarterLength)*5)
            elif isinstance(part, music21.chord.Chord):
                for note in part.notes:
                    notes.append(note.pitch.nameWithOctave)
                    offsets.append(offset)
                    durations.append(
                        self._convert_to_float(note.duration.quarterLength)*5
                    )
            prev_offset = current_offset
        return notes, offsets, durations

    def _load_data(self):
        """
        Load the MIDI data files from `self.dataset_dir` folder
        and parse the files to generate the processed dataset
        """
        if not os.path.exists(self.dataset_dir):
            logger.error(f"{self.dataset_dir} does not exist")
            exit(1)
        data = []

        print(f"Processing MIDI: {self.dataset_dir}")
        midi_files = os.listdir(self.dataset_dir)
        for midi_file in midi_files:
            midi = music21.converter.parse(os.path.join(self.dataset_dir, midi_file))
            notes, offsets, durations = self._process_midi(midi)
            data.append(
                {
                    "notes": notes,
                    "offsets": offsets,
                    "durations": durations,
                }
            )
        all_notes = [note for sample in data for note in sample["notes"]]
        vocab = self._generate_vocab(all_notes)
        return data, vocab

    def _generate_vocab(self, notes):
        """
        Parse the notes to generate a vocab dictionary of vocab_id and the note
        """
        notes = sorted(set(note for note in notes))
        vocab = {note: vocab_id for vocab_id, note in enumerate(notes)}
        return vocab

    def _preprocess(self, data):
        """
        Preprocess the data to generate inputs and targets
        """
        samples = []
        for sample in data:
            for i in range(0, len(sample["notes"]), CHORD_LENGTH):
                notes = sample["notes"][i : i + CHORD_LENGTH]
                offsets = sample["offsets"][i : i + CHORD_LENGTH]
                durations = sample["durations"][i : i + CHORD_LENGTH]
                if len(notes) < CHORD_LENGTH:
                    notes += ["R"] * (CHORD_LENGTH - len(notes))
                    offsets += [offsets[-1] + 1.0] * (CHORD_LENGTH - len(offsets))
                    durations += [0] * (CHORD_LENGTH - len(durations))
                samples.append(
                    {
                        "notes": notes,
                        "offsets": offsets,
                        "durations": durations,
                    }
                )
        return samples

    def __getitem__(self, index):
        """
        Return a sample at given index from the dataset
        """
        notes = torch.tensor(
            [self.vocab[note] for note in self.dataset[index]["notes"]],
            dtype=torch.long,
        )
        offsets = torch.tensor(self.dataset[index]["offsets"])
        durations = torch.tensor(self.dataset[index]["durations"])
        return notes, offsets, durations

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.dataset)

    def get_notes(self, melody):
        """
        Map the note_ids to the notes
        """
        reverse_vocab = {vocab_id: vocab for (vocab, vocab_id) in self.vocab.items()}
        notes = [reverse_vocab[note] for note in melody]
        return notes
