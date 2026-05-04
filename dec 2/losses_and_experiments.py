import torch
import torchaudio
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dataset import AudioDataset
from audio import NeuralAudioCodec

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(SCRIPT_DIR, "neural_codec_final.pt")   

model = NeuralAudioCodec(num_quantizers=4, codebook_size=512, embed_dim=128, hidden_dim=512).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

dataset = AudioDataset("./dev-clean")

for idx in [0, 1, 2]:
    audio, _ = dataset[idx]
    input_audio = audio.unsqueeze(0).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        x_hat, _, _ = model(input_audio)

        x_hat_de = torchaudio.functional.deemphasis(
            x_hat.squeeze(1), coeff=0.85
        ).unsqueeze(1)
        original_de = torchaudio.functional.deemphasis(
            input_audio.squeeze(1), coeff=0.85
        ).unsqueeze(1)

    torchaudio.save(f"original_{idx}.wav", original_de.squeeze(0).cpu(), 16000)
    torchaudio.save(f"reconstructed_{idx}.wav", x_hat_de.squeeze(0).cpu(), 16000)
    print(f"Saved sample {idx} with de-emphasis")
print("Готово!")