import json
import os
import numpy as np
from music21 import stream, note, chord


MODEL_FILE = "output/music_model.npz"
VOCAB_FILE = "output/vocabulary.json"
DATA_FILE = "data/notes.json"
OUTPUT_FILE = "output/generated_music.mid"

SEQUENCE_LENGTH = 30
GENERATED_NOTES = 160

TEMPERATURE = 1.7
TOP_K = 15
TOP_P = 0.90

REPETITION_PENALTY = 1.15
RECENT_NOTES_TO_CHECK = 12


def load_model():
    model = np.load(MODEL_FILE)

    with open(VOCAB_FILE, "r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    note_to_index = vocabulary["note_to_index"]
    index_to_note = vocabulary["index_to_note"]

    return model, note_to_index, index_to_note


def load_notes():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        notes = data

    elif isinstance(data, dict) and "notes" in data:
        notes = data["notes"]

    elif isinstance(data, dict) and "events" in data:
        notes = data["events"]

    else:
        notes = list(data.values())

    cleaned = []

    for item in notes:
        if isinstance(item, str):
            cleaned.append(item)

        elif isinstance(item, dict):
            if "note" in item:
                cleaned.append(str(item["note"]))

            elif "pitch" in item:
                cleaned.append(str(item["pitch"]))

    if len(cleaned) < SEQUENCE_LENGTH:
        raise ValueError("Not enough musical events available.")

    return cleaned


def select_random_seed(notes):
    max_start = len(notes) - SEQUENCE_LENGTH

    start_position = np.random.randint(0, max_start + 1)

    seed = notes[
        start_position:start_position + SEQUENCE_LENGTH
    ]

    print(f"Random seed position: {start_position}")

    return seed


def softmax(values):
    values = values - np.max(values)

    probabilities = np.exp(values)

    total = np.sum(probabilities)

    if total == 0:
        return np.ones(len(values)) / len(values)

    return probabilities / total


def apply_repetition_penalty(
    probabilities,
    generated,
    note_to_index
):
    adjusted = probabilities.copy()

    recent_notes = generated[-RECENT_NOTES_TO_CHECK:]

    for musical_note in recent_notes:

        if musical_note not in note_to_index:
            continue

        index = note_to_index[musical_note]

        adjusted[index] /= REPETITION_PENALTY

    total = np.sum(adjusted)

    if total > 0:
        adjusted /= total

    return adjusted


def top_k_top_p_sampling(probabilities):
    sorted_indices = np.argsort(probabilities)[::-1]

    if TOP_K > 0:
        sorted_indices = sorted_indices[:TOP_K]

    selected_probabilities = probabilities[sorted_indices]

    sorted_probabilities = selected_probabilities[
        np.argsort(selected_probabilities)[::-1]
    ]

    sorted_indices = sorted_indices[
        np.argsort(selected_probabilities)[::-1]
    ]

    cumulative_probability = np.cumsum(sorted_probabilities)

    keep_mask = cumulative_probability <= TOP_P

    if not np.any(keep_mask):
        keep_mask[0] = True

    filtered_indices = sorted_indices[keep_mask]
    filtered_probabilities = sorted_probabilities[keep_mask]

    total = np.sum(filtered_probabilities)

    if total == 0:
        filtered_probabilities = np.ones(
            len(filtered_indices)
        ) / len(filtered_indices)

    else:
        filtered_probabilities /= total

    selected_index = np.random.choice(
        filtered_indices,
        p=filtered_probabilities
    )

    return selected_index


def generate_notes(
    model,
    note_to_index,
    index_to_note,
    seed
):
    Wxh = model["Wxh"]
    Whh = model["Whh"]
    Why = model["Why"]
    bh = model["bh"]
    by = model["by"]

    hidden_size = Wxh.shape[0]
    vocabulary_size = Wxh.shape[1]

    generated = list(seed)

    current_sequence = list(seed)

    for step in range(GENERATED_NOTES):

        hidden = np.zeros(hidden_size)

        for current_note in current_sequence:

            if current_note not in note_to_index:
                continue

            x = np.zeros(vocabulary_size)

            index = note_to_index[current_note]

            x[index] = 1.0

            hidden = np.tanh(
                Wxh @ x +
                Whh @ hidden +
                bh
            )

        logits = Why @ hidden + by

        probabilities = softmax(
            logits / TEMPERATURE
        )

        probabilities = apply_repetition_penalty(
            probabilities,
            generated,
            note_to_index
        )

        next_index = top_k_top_p_sampling(
            probabilities
        )

        next_note = index_to_note[
            str(next_index)
        ]

        generated.append(next_note)

        current_sequence.append(next_note)

        if len(current_sequence) > SEQUENCE_LENGTH:
            current_sequence.pop(0)

    return generated


def convert_to_midi(generated_notes):

    music = stream.Stream()

    for pitch_name in generated_notes:

        try:

            if pitch_name.lower() == "rest":

                music.append(
                    note.Rest(
                        quarterLength=1
                    )
                )

            elif "." in pitch_name and "+" in pitch_name:

                chord_notes = pitch_name.split("+")

                new_chord = chord.Chord(
                    chord_notes
                )

                new_chord.quarterLength = 1

                music.append(new_chord)

            else:

                new_note = note.Note(
                    pitch_name
                )

                new_note.quarterLength = 1

                music.append(new_note)

        except Exception:
            continue

    os.makedirs(
        "output",
        exist_ok=True
    )

    music.write(
        "midi",
        fp=OUTPUT_FILE
    )

    return OUTPUT_FILE


def generate_music():

    print("Loading trained RNN model...")

    model, note_to_index, index_to_note = load_model()

    print("Loading musical dataset...")

    notes = load_notes()

    print("Selecting random seed sequence...")

    seed = select_random_seed(notes)

    print(
        f"Seed length: {len(seed)}"
    )

    print(
        f"Generating {GENERATED_NOTES} new musical events..."
    )

    print(
        f"Temperature: {TEMPERATURE}"
    )

    print(
        f"Top-K sampling: {TOP_K}"
    )

    print(
        f"Top-P sampling: {TOP_P}"
    )

    generated = generate_notes(
        model,
        note_to_index,
        index_to_note,
        seed
    )

    print("Converting generated sequence to MIDI...")

    output_file = convert_to_midi(
        generated
    )

    print()
    print("Music generation complete.")
    print(
        f"Generated MIDI saved to: {output_file}"
    )


if __name__ == "__main__":
    generate_music()