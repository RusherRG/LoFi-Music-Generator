import music21
import numpy

from dataset import LoFiMIDIDataset
from models.lstm_generate import lstm_generate

NOTE_TYPE = {"eighth": 0.5, "quarter": 1, "half": 2, "16th": 0.25}


def create_midi(generated_notes):
    offset = 0
    midi_notes = []

    for note in generated_notes:
        curr_type = numpy.random.choice(
            list(NOTE_TYPE.keys()), p=[0.65, 0.05, 0.05, 0.25]
        )
        # chord
        if note.find(".") != -1:
            chord_notes = []
            for current_note in note.split("."):
                new_note = music21.note.Note(int(current_note))
                new_note.storedInstrument = music21.instrument.BassDrum()
                chord_notes.append(new_note)
            new_chord = music21.chord.Chord(chord_notes, type=curr_type)
            new_chord.offset = offset
            midi_notes.append(new_chord)
        elif note == "R":
            curr_type = "16th"
            new_rest = music21.note.Rest(type=curr_type)
            new_rest.offset = offset
            midi_notes.append(new_rest)
        # pattern is a note
        else:
            new_note = music21.note.Note(str(note), type=curr_type)
            new_note.offset = offset
            new_note.storedInstrument = music21.instrument.BassDrum()
            midi_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += NOTE_TYPE[curr_type]

    midi_stream = music21.stream.Stream(midi_notes)
    midi_stream.write("midi", fp="test_output.mid")


def generate(model_name: str, dataset_dir: str = "./dataset/"):
    dataset = LoFiMIDIDataset(dataset_dir=dataset_dir)

    if model_name == "lstm":
        generated_melody = lstm_generate(dataset)
        generated_notes = dataset.get_notes(generated_melody)

    create_midi(generated_notes)


if __name__ == "__main__":
    generate("lstm", "./dataset")
