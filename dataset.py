import os
import pickle

import music21
from torch.utils.data import Dataset
from models.constants import *


class LoFiMIDIDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.data, self.vocab = self._load_data()
        self.vocab_size = len(self.vocab)

        self.dataset = self._preprocess(self.data)

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

        for part in notesAndRests:
            if isinstance(part, music21.note.Note):
                notes.append(part.pitch.nameWithOctave)
            elif isinstance(part, music21.note.Rest):
                notes.append("R")
            elif isinstance(part, music21.chord.Chord):
                notes.append(".".join(p.nameWithOctave for p in part.notes))

        return notes

    def _load_data(self):
        """
        Load the MIDI data files from `self.dataset_dir` folder
        and parse the files to generate the processed dataset
        """
        if not os.path.exists(self.dataset_dir):
            logger.error(f"{self.dataset_dir} does not exist")
            exit(1)
        data = []
        midi_files = os.listdir(self.dataset_dir)
        for midi_file in midi_files:
            print(f"Processing MIDI: {midi_file}")
            midi = music21.converter.parse(os.path.join(self.dataset_dir, midi_file))
            notes = self._process_midi(midi)
            data.append(notes)

        all_notes = [note for notes in data for note in notes]
        vocab = self._generate_vocab(all_notes)
        return data, vocab

    def _generate_vocab(self, notes):
        """
        Parse the notes to generate a vocab dictionary of vocab_id and the note
        """
        notes = set(notes.sort())
        vocab = {note: vocab_id for vocab_id, note in enumerate(notes)}
        return vocab

    def _preprocess(self, data):
        """
        Preprocess the data to generate inputs and targets
        """
        inputs_targets = []
        for sample in data:
            for i in range(0, len(sample), MELODY_LENGTH):
                sequence = sample[i : i + MELODY_LENGTH]
                sequence_ids = [self.vocab[s] for s in sequence]
                inputs_targets.append((sequence_ids[:-1], sequence_ids[-1]))
        return inputs_targets

    def __getitem__(self, index):
        """
        Return a sample at given index from the dataset
        """
        return self.dataset[index]

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.dataset)
