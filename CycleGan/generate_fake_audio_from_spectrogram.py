import os
import sys
import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path

# === Add CycleGAN project path ===
sys.path.append(r"C:\Users\morel\PycharmProjects\PythonProject10\CycleGan\pytorch-CycleGAN-for-audio-master")

from options.test_options import TestOptions
from data import create_dataset
from models import create_model

# === Define parameters ===
sys.argv = [
    '',
    '--dataroot', r'C:\Users\morel\PycharmProjects\PythonProject10\datasets\dog_cat\dc',
    '--name', 'dc_model',
    '--model', 'cycle_gan',
    '--dataset_mode', 'unaligned_spec',
    '--direction', 'BtoA',
    '--checkpoints_dir', r'C:\Users\morel\PycharmProjects\PythonProject10\datasets\dog_cat',
    '--gpu_ids', '-1',
    '--no_dropout',
    '--num_test', '1000',
    '--results_dir', r'C:\Users\morel\PycharmProjects\PythonProject10\datasets\dog_cat\results',
    '--phase', 'test',
    '--serial_batches',
    '--no_flip',
    '--batch_size', '1',
    '--num_threads', '0',
    '--preprocess', 'none',
    '--input_nc', '1',
    '--output_nc', '1',
]

opt = TestOptions().parse()
opt.eval = True

dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()

# === Create output directory ===
output_dir = os.path.join(opt.results_dir, opt.name + f"_{opt.direction}")
os.makedirs(output_dir, exist_ok=True)

# === Save spectrogram and audio with length matching ===
def save_spectrogram_and_audio(spec, filename_prefix, target_length):
    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec.squeeze(), sr=16000, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}.png")
    plt.close()

    # Reconstruct waveform
    mag = np.clip(spec.squeeze(), 0, 1)
    wav = librosa.griffinlim(mag * 10)

    # Match length
    if len(wav) > target_length:
        wav = wav[:target_length]
    elif len(wav) < target_length:
        wav = np.pad(wav, (0, target_length - len(wav)))

    sf.write(f"{filename_prefix}.wav", wav, 16000)

# === Process files ===
for i, data in enumerate(dataset):
    if i >= opt.num_test:
        break

    model.set_input(data)
    model.test()

    fake_B = model.fake_B.detach().cpu().numpy()
    fake_A = model.fake_A.detach().cpu().numpy()

    # Get original file path
    input_path = data['A_paths'][0] if opt.direction == 'AtoB' else data['B_paths'][0]
    base_name = Path(input_path).stem
    print(f"[{i}] Processing {base_name}...")

    # Load original waveform to get length
    if input_path.endswith('.npy'):
        wav_candidate = input_path.replace('.npy', '')
    else:
        wav_candidate = os.path.splitext(input_path)[0]

    if os.path.exists(wav_candidate + '.wav'):
        original_audio, _ = librosa.load(wav_candidate + '.wav', sr=16000)
    elif os.path.exists(wav_candidate + '.mp3'):
        original_audio, _ = librosa.load(wav_candidate + '.mp3', sr=16000)
    else:
        print(f"Warning: No original audio found for {base_name}, skipping.")
        continue

    target_length = len(original_audio)

    # Save outputs
    np.save(os.path.join(output_dir, f"{base_name}_fake_B.npy"), fake_B)
    np.save(os.path.join(output_dir, f"{base_name}_fake_A.npy"), fake_A)

    save_spectrogram_and_audio(fake_B, os.path.join(output_dir, f"{base_name}_fake_B"), target_length)
    save_spectrogram_and_audio(fake_A, os.path.join(output_dir, f"{base_name}_fake_A"), target_length)

print("All fake_B and fake_A saved with matching names and lengths.")
