import json
import os
import numpy as np

DATA_FILE = "data/notes.json"
MODEL_FILE = "output/music_rnn.npz"
VOCAB_FILE = "output/vocabulary.json"

SEQUENCE_LENGTH = 30
HIDDEN_SIZE = 64
LEARNING_RATE = 0.005
EPOCHS = 20


def load_notes():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Could not find {DATA_FILE}")

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        notes = data
    elif isinstance(data, dict):
        if "notes" in data:
            notes = data["notes"]
        elif "events" in data:
            notes = data["events"]
        else:
            notes = list(data.values())
    else:
        raise ValueError("Unsupported notes.json format.")

    cleaned_notes = []

    for note in notes:
        if isinstance(note, str):
            cleaned_notes.append(note)
        elif isinstance(note, dict):
            if "note" in note:
                cleaned_notes.append(str(note["note"]))
            elif "pitch" in note:
                cleaned_notes.append(str(note["pitch"]))

    if len(cleaned_notes) < SEQUENCE_LENGTH + 1:
        raise ValueError("Not enough musical events to train the model.")

    return cleaned_notes


def create_training_data(notes):
    vocabulary = sorted(set(notes))

    note_to_index = {
        note: index for index, note in enumerate(vocabulary)
    }

    index_to_note = {
        str(index): note for note, index in note_to_index.items()
    }

    sequences = []
    targets = []

    for i in range(len(notes) - SEQUENCE_LENGTH):
        sequence = notes[i:i + SEQUENCE_LENGTH]
        target = notes[i + SEQUENCE_LENGTH]

        sequences.append([
            note_to_index[note]
            for note in sequence
        ])

        targets.append(note_to_index[target])

    return sequences, targets, vocabulary, note_to_index, index_to_note


def softmax(values):
    values = values - np.max(values)
    probabilities = np.exp(values)
    return probabilities / np.sum(probabilities)


def train_model():
    print("Loading musical data...")

    notes = load_notes()

    print(f"Total musical events: {len(notes)}")

    sequences, targets, vocabulary, note_to_index, index_to_note = (
        create_training_data(notes)
    )

    vocab_size = len(vocabulary)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Training sequences: {len(sequences)}")
    print("Starting NumPy RNN training...")

    rng = np.random.default_rng(42)

    Wxh = rng.normal(
        0,
        0.01,
        (HIDDEN_SIZE, vocab_size)
    )

    Whh = rng.normal(
        0,
        0.01,
        (HIDDEN_SIZE, HIDDEN_SIZE)
    )

    Why = rng.normal(
        0,
        0.01,
        (vocab_size, HIDDEN_SIZE)
    )

    bh = np.zeros(HIDDEN_SIZE)
    by = np.zeros(vocab_size)

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for sequence, target in zip(sequences, targets):
            hidden_states = []
            inputs = []

            h = np.zeros(HIDDEN_SIZE)

            for note_index in sequence:
                x = np.zeros(vocab_size)
                x[note_index] = 1.0

                h = np.tanh(
                    Wxh @ x +
                    Whh @ h +
                    bh
                )

                inputs.append(x)
                hidden_states.append(h.copy())

            logits = Why @ h + by
            probabilities = softmax(logits)

            loss = -np.log(
                probabilities[target] + 1e-8
            )

            total_loss += loss

            dlogits = probabilities.copy()
            dlogits[target] -= 1.0

            dWhy = np.outer(
                dlogits,
                h
            )

            dby = dlogits.copy()

            dWxh = np.zeros_like(Wxh)
            dWhh = np.zeros_like(Whh)
            dbh = np.zeros_like(bh)

            dh = Why.T @ dlogits

            for t in reversed(range(SEQUENCE_LENGTH)):
                current_h = hidden_states[t]

                if t > 0:
                    previous_h = hidden_states[t - 1]
                else:
                    previous_h = np.zeros(HIDDEN_SIZE)

                dtanh = (
                    1.0 - current_h ** 2
                ) * dh

                dWxh += np.outer(
                    dtanh,
                    inputs[t]
                )

                dWhh += np.outer(
                    dtanh,
                    previous_h
                )

                dbh += dtanh

                dh = Whh.T @ dtanh

            for gradient in [
                dWxh,
                dWhh,
                dWhy,
                dbh,
                dby
            ]:
                np.clip(
                    gradient,
                    -5,
                    5,
                    out=gradient
                )

            Wxh -= LEARNING_RATE * dWxh
            Whh -= LEARNING_RATE * dWhh
            Why -= LEARNING_RATE * dWhy
            bh -= LEARNING_RATE * dbh
            by -= LEARNING_RATE * dby

        average_loss = total_loss / len(sequences)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {average_loss:.4f}"
        )

    os.makedirs("output", exist_ok=True)

    np.savez(
        MODEL_FILE,
        Wxh=Wxh,
        Whh=Whh,
        Why=Why,
        bh=bh,
        by=by
    )

    with open(VOCAB_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {
                "note_to_index": note_to_index,
                "index_to_note": index_to_note
            },
            file,
            indent=2
        )

    print()
    print("Training complete.")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Vocabulary saved to: {VOCAB_FILE}")


if __name__ == "__main__":
    train_model()