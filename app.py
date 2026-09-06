import os
import subprocess
import streamlit as st

MODEL_FILE = "output/music_model.npz"
MIDI_FILE = "output/generated_music.mid"

st.set_page_config(
    page_title="AI Music Generator",
    page_icon="Music",
    layout="centered"
)

st.title("AI Music Generator")
st.write(
    "Generate a new musical sequence using an RNN trained on MIDI data."
)

st.divider()

st.subheader("Model Status")

if os.path.exists(MODEL_FILE):
    st.success("Trained RNN model is ready.")
else:
    st.warning("No trained model found. Train the model first.")

st.divider()

st.subheader("Generate Music")

if st.button("Generate New Music", use_container_width=True):
    if not os.path.exists(MODEL_FILE):
        st.error("Please train the model before generating music.")
    else:
        with st.spinner("Generating music..."):
            result = subprocess.run(
                ["python", "generate.py"],
                capture_output=True,
                text=True
            )

        if result.returncode == 0 and os.path.exists(MIDI_FILE):
            st.success("New music generated successfully.")

            st.audio(
                MIDI_FILE,
                format="audio/midi"
            )

            with open(MIDI_FILE, "rb") as file:
                st.download_button(
                    label="Download Generated MIDI",
                    data=file,
                    file_name="generated_music.mid",
                    mime="audio/midi",
                    use_container_width=True
                )
        else:
            st.error("Music generation failed.")
            st.code(result.stderr)

st.divider()

st.subheader("How It Works")

st.write(
    "1. MIDI music data is processed with music21."
)

st.write(
    "2. Musical notes are converted into sequences."
)

st.write(
    "3. A NumPy-based recurrent neural network learns patterns "
    "from the sequences."
)

st.write(
    "4. The trained RNN generates new musical events."
)

st.write(
    "5. The generated events are converted back into a MIDI file."
)