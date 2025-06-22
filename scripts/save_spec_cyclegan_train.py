import librosa
import numpy as np
import os
import time
from glob import glob

def compute_spec(y):
    n_fft = 320
    win_length = n_fft
    hop_length = int(n_fft / 2)
    window = 'hamming'
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    data, phase = librosa.magphase(D)

    data = np.log1p(data)  # log compression
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        print("⚠️  STD is zero — skipping")
        return None
    data = (data - mean) / std
    data = data / np.abs(data).max()

    print(f'mean={mean:.4f}, std={std:.4f}, shape={data.shape}')
    return data.reshape(1, data.shape[0], data.shape[1])

# === הגדרות ===
in_len = 16000 * 5  # 5 שניות בדיוק
base_output_dir = r"C:\Users\morel\PycharmProjects\PythonProject10\datasets\dog_cat"

species_dict = {
    'A': r"C:\Users\morel\PycharmProjects\PythonProject10\datasets\dog_cat\dog",
    'B': r"C:\Users\morel\PycharmProjects\PythonProject10\datasets\dog_cat\cat"
}

start = time.time()

for label, src_folder in species_dict.items():
    train_path = os.path.join(base_output_dir, f"train{label}")
    test_path = os.path.join(base_output_dir, f"test{label}")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    audio_files = glob(os.path.join(src_folder, '*.wav'))
    print(f"\n📂 Processing {len(audio_files)} files from {src_folder}")

    for idx, audio_path in enumerate(audio_files):
        y, sr = librosa.load(audio_path, sr=16000)
        print(f"→ {os.path.basename(audio_path)}: original length = {len(y)} samples")

        # התאמה לאורך 5 שניות
        if len(y) < in_len:
            pad_len = in_len - len(y)
            y = np.pad(y, (0, pad_len), mode='constant')
            print(f"🧵 Padded to 5 seconds ({in_len} samples)")
        elif len(y) > in_len:
            y = y[:in_len]
            print(f"✂️ Trimmed to 5 seconds ({in_len} samples)")

        spec = compute_spec(y)
        if spec is None or np.isnan(spec).any():
            print(f"❌ Skipping {audio_path} due to invalid spec")
            continue

        base_name = os.path.splitext(os.path.basename(audio_path))[0] + '.npy'
        save_path = os.path.join(train_path if idx < 0.9 * len(audio_files) else test_path, base_name)

        print(f"✅ Saving to {save_path}")
        np.save(save_path, spec)

end = time.time()
print(f"\n✅ Processing complete in {(end - start)/60:.2f} minutes.")
