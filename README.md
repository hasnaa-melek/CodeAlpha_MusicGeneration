# CodeAlpha Music Generation with AI

An AI-based music generation project developed as part of the CodeAlpha Artificial Intelligence Internship.

The project uses MIDI music data, the `music21` library for musical data processing, and a NumPy-based Recurrent Neural Network (RNN) to learn musical patterns and generate new MIDI music.

## Project Overview

This project demonstrates an end-to-end AI music generation pipeline:

1. Collect MIDI music data.
2. Process MIDI files using `music21`.
3. Convert musical events into numerical sequences.
4. Train a Recurrent Neural Network using NumPy.
5. Generate new musical sequences using the trained model.
6. Convert the generated sequences back into MIDI format.
7. Provide a simple Streamlit interface for generating music.

## Technologies Used

- Python
- NumPy
- music21
- Streamlit
- Recurrent Neural Network (RNN)
- MIDI

## Features

- MIDI data preprocessing
- Musical event extraction
- RNN-based music generation
- Random seed selection for different generations
- Temperature-based sampling
- Top-K sampling
- Top-P sampling
- Repetition penalty
- MIDI output generation
- Streamlit web interface

## Project Structure

```text
CodeAlpha_MusicGeneration/
│
├── data/
│   ├── midi/
│   └── notes.json
│
├── output/
│   ├── generated_music.mid
│   ├── music_model.npz
│   └── vocabulary.json
│
├── app.py
├── collect_data.py
├── preprocess.py
├── train.py
├── generate.py
├── requirements.txt
└── README.md
