from pathlib import Path
from music21 import corpus


OUTPUT_DIR = Path("data/midi")
MAX_FILES = 30


def collect_bach_midi():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bach_files = list(corpus.search("bach"))

    print(f"Found {len(bach_files)} Bach files in the music21 corpus.")

    saved = 0

    for index, file_info in enumerate(bach_files):
        if saved >= MAX_FILES:
            break

        try:
            score = file_info.parse()

            output_path = OUTPUT_DIR / f"bach_{saved + 1}.mid"
            score.write("midi", fp=str(output_path))

            saved += 1
            print(f"Saved: {output_path}")

        except Exception as error:
            print(f"Skipped file {index + 1}: {error}")

    print(f"\nFinished. {saved} MIDI files were created.")


if __name__ == "__main__":
    collect_bach_midi()