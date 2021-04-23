import nussl
import soundfile as sf

from pathlib import Path
from tqdm import tqdm


def separation(path: Path):
    history = nussl.AudioSignal(path)
    repet = nussl.separation.primitive.Repet(history)
    estimated = repet()

    foreground = estimated[1].audio_data[0]
    background = estimated[0].audio_data[0]
    return foreground, background


def save_audio(path: Path, audio, samplerate=32000):
    sf.write(path, audio, samplerate=samplerate)


if __name__ == '__main__':
    audio_dir = Path("./train_soundscapes")
    foreground_dir = Path("./train_foreground")
    background_dir = Path("./train_background")

    foreground_dir.mkdir(exist_ok=True, parents=True)
    background_dir.mkdir(exist_ok=True, parents=True)

    audio_paths = list(audio_dir.glob("*.ogg"))
    for audio_path in tqdm(audio_paths):
        foreground, background = separation(audio_path)
        foreground_path = foreground_dir / \
            audio_path.name.replace(".ogg", ".wav")
        background_path = background_dir / \
            audio_path.name.replace(".ogg", ".wav")
        save_audio(foreground_path, foreground, samplerate=32000)
        save_audio(background_path, background, samplerate=32000)
