from pathlib import Path
import json

from music21 import chord, converter, note


MIDI_DIR = Path("data/midi")
OUTPUT_FILE = Path("data/notes.json")


def extract_notes_from_midi(file_path):
    score = converter.parse(file_path)
    tokens = []

    for element in score.flatten().notes:

        if isinstance(element, note.Note):
            tokens.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            chord_token = ".".join(str(pitch) for pitch in element.pitches)
            tokens.append(chord_token)

    return tokens


def preprocess_dataset():
    all_notes = []

    midi_files = list(MIDI_DIR.glob("*.mid"))

    print(f"Found {len(midi_files)} MIDI files.")

    for index, midi_file in enumerate(midi_files, start=1):
        try:
            notes = extract_notes_from_midi(midi_file)
            all_notes.extend(notes)

            print(
                f"Processed {index}/{len(midi_files)}: "
                f"{midi_file.name} - {len(notes)} notes"
            )

        except Exception as error:
            print(f"Skipped {midi_file.name}: {error}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(all_notes, file)

    print(f"\nTotal musical events: {len(all_notes)}")
    print(f"Saved processed data to: {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess_dataset()