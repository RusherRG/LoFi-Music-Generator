import music21
import numpy

from dataset import LoFiMIDIDataset
from models.lstm_generate import lstm_generate
from models.vae_generate import vae_generate

NOTE_TYPE = {"eighth": 0.5, "quarter": 1, "half": 2, "16th": 0.25}


def create_midi(generated_notes, offsets, durations):
    """
    Creating the MIDI file using music21 library to create the music elements
    like music21.Chord, music21.Note and stitch them to generate a sample
    """
    current_offset = 0
    midi_notes = []
    print("Creating MIDI")
    for note, offset, duration in zip(generated_notes, offsets, durations):
        current_offset += offset / 5
        # it is a rest note
        if note == "R":
            new_rest = music21.note.Rest()
            new_rest.offset = current_offset
            new_rest.quarterLength = duration / 5
            midi_notes.append(new_rest)
        # it is a note
        else:
            new_note = music21.note.Note(str(note))
            new_note.offset = current_offset
            new_note.quarterLength = duration / 5
            new_note.storedInstrument = music21.instrument.Piano()
            midi_notes.append(new_note)
    # generate the MIDI stream and save to file
    midi_stream = music21.stream.Stream(midi_notes)
    midi_stream.write("midi", fp="test_output.mid")


def generate(model_name: str, dataset_dir: str = "./dataset/"):
    """
    Run generation to generate new MIDI samples using the trained model
    """
    dataset = LoFiMIDIDataset(dataset_dir=dataset_dir)

    # check the model name
    if model_name == "lstm":
        # generate melody and extract notes
        generated_melody = lstm_generate(dataset)
        generated_notes = dataset.get_notes(generated_melody)

    elif model_name == "vae":
        # generate melody and extract notes
        notes, offsets, durations = vae_generate(dataset)
        generated_notes = dataset.get_notes(notes)

    # create the MIDI sample based on the generated samples
    create_midi(generated_notes, offsets, durations)


if __name__ == "__main__":
    generate("vae", "./dataset")
