import os
import glob
import librosa
import warnings
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from tqdm import trange
from mutagen.mp3 import MP3

# Suppress soundfile warning on loading MP3
warnings.simplefilter('ignore')
DOWNLOAD_DIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files/audio_200k'
SPEC_DIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files/spec_200k_2'
MAX_CHUNKS_PER_LABEL = 50

# Check if the directories exist
if not os.path.exists(SPEC_DIR):
    os.makedirs(SPEC_DIR)

SAMPLING_FREQUENCY = 16000
NFTT = 1024
HOP_LENGTH = 512
CHUNK_PER_SOUND = 5
CHUNK_WIDTH = 128

def get_melspec(spec, n_mels):
    # Power spectrum
    powerspec = np.abs(spec)**2
    melspec = librosa.feature.melspectrogram(S=powerspec, n_mels=n_mels)
    S = librosa.power_to_db(melspec, np.max)
    return S

def get_spectrum(path, n_fft, hop_length, sr=None):
    try:
        y, sr = librosa.load(path, sr=sr)
    except:
        y, o_sr = sf.read(path)
        y = librosa.core.resample(y, o_sr, sr)
    y_stft = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length)
    return y_stft

def compute_spectrogram(path, sr=SAMPLING_FREQUENCY, n_mels=128):
    y_stft = get_spectrum(path, n_fft=NFTT, sr=sr, hop_length=HOP_LENGTH)
    melspec = get_melspec(y_stft, n_mels=n_mels)
    # Convert melspec from -80 to 0dB to range [0,1]
    spec = (melspec + 80.0) / 80.0
    return spec

def spec_to_chunks(path, remove_silence=False):
    spec = compute_spectrogram(path)
    spec_length = spec.shape[1]
    output_chunks = []
    # print(f'Spec // chunk width = {spec_length} // {CHUNK_WIDTH} = {spec_length // CHUNK_WIDTH}')
    for i in range(0, spec_length // CHUNK_WIDTH):
        chunk = spec[:, i * CHUNK_WIDTH : (i + 1) * CHUNK_WIDTH]
        if chunk.shape[1] != CHUNK_WIDTH:
            continue
        if not has_enough_energy(chunk, threshold=2500):
            continue

        output_chunks.append(chunk)
    return output_chunks

def save_chucks(original_path, chunks):
    sound_id = Path(original_path).stem

    if not os.path.exists(f'{SPEC_DIR}'):
        os.makedirs(f'{SPEC_DIR}')
    assert len(glob.glob(f'{SPEC_DIR}/{sound_id}_*.npy')) == 0, f'{sound_id} already has specs.'

    for i, chunk in enumerate(chunks):
        chunk_path = f'{SPEC_DIR}/{sound_id}_{i + 1:06d}.npy'
        np.save(chunk_path, chunk)

    return len(chunks)

def has_enough_energy(spec, threshold=2500):
    return get_spec_energy(spec) > threshold

def get_spec_energy(spec):
    return np.sum(spec ** 2)

def visualize_chunks(path, chunks):
    plt.figure(figsize=(6, 8))
    plt.tight_layout()
    print(f'Path: {path} | # of Chunks : {len(chunks)}')

    for i, chunk in enumerate(chunks[:16]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(chunk)
        plt.xlabel(f'{get_spec_energy(chunk):.2f}')

    # plt.title(path, y=-1)
    plt.show()

def audio_to_chunk(path, sr=SAMPLING_FREQUENCY):
    # (Feature size) = (sampling rate) x (duration) / (Hop size)
    # (Sample per chunk) = (Chunk width) x (Hop size)
    samples_per_chunk = CHUNK_WIDTH * HOP_LENGTH
    y, sr = librosa.load(path, sr=SAMPLING_FREQUENCY)
    audio_chunks = []

    for i in range(0, len(y) // samples_per_chunk):
        # Approximating waveforms of corresponding chunked spectrogram
        audio = y[i * samples_per_chunk : (i + 1) * samples_per_chunk]
        if len(audio) != samples_per_chunk:
            continue
        audio_chunks.append(audio)
    return audio_chunks

def projection_setup(path, audio_dir, spec_dir):
    # print(f'Setup for projection: Chunking {path}')
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    if not os.path.exists(spec_dir):
        os.makedirs(spec_dir)
    spec_chunks = spec_to_chunks(path)
    audio_chunks = audio_to_chunk(path)
    audio_nums = []

    for spec_chunk, audio_chunk in zip(spec_chunks, audio_chunks):
        spec_existing = len(glob.glob(f'{spec_dir}/*.npy'))
        audio_existing = len(glob.glob(f'{audio_dir}/*.wav'))
        assert spec_existing == audio_existing, (path, audio_dir, spec_dir)
        spec_path = f'{spec_dir}/{spec_existing + 1:06d}.npy'
        audio_path = f'{audio_dir}/{audio_existing + 1:06d}.wav'
        audio_nums.append(f'{audio_existing + 1:06d}.wav')
        np.save(spec_path, spec_chunk)
        librosa.output.write_wav(audio_path, audio_chunk, SAMPLING_FREQUENCY)

    return spec_chunks, audio_nums

if __name__ == '__main__':
    audio_files = glob.glob(f'{DOWNLOAD_DIR}/*.mp3')
    pbar = trange(len(audio_files))
    error_count = 0

    for i in pbar:
        path = audio_files[i]
        try:
            chunks = spec_to_chunks(path, remove_silence=True)
            save_chucks(path, chunks)
            pbar.set_description(f'{len(chunks)} added')
            # visualize_chunks(path, chunks)
        except Exception as e:
            error_count += 1
            print(f'Error #{error_count}: {e}')