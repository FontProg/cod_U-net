import torch
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from timm import create_model
import os

SAVE_DIR = "results_new_model"
AUDIO_DIR = "new_test_model/data_new_model/val_split"
CSV_PATH = "new_test_model/data_new_model/val_split/val_annotations.csv"
MODEL_PATH = os.path.join(SAVE_DIR, "best_efficientnet_b4.pt")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_MELS = 128
HOP_LENGTH = 512
TARGET_FRAMES = 380

# функции от модели
def load_audio(filepath, sr=32000):
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    return y

def process_length(y, target_frames=TARGET_FRAMES, hop_length=HOP_LENGTH):
    target_length = target_frames * hop_length
    if len(y) < target_length:
        n_repeat = int(np.ceil(target_length / len(y)))
        y = np.tile(y, n_repeat)[:target_length]
    else:
        y = y[:target_length]
    return y

def create_mel_spectrogram(y, sr=32000, n_mels=N_MELS, hop_length=HOP_LENGTH):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    if mel_db.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :TARGET_FRAMES]
    return mel_db

df = pd.read_csv(CSV_PATH)
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])
NUM_CLASSES = len(label_encoder.classes_)

class EfficientNetAudio(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = create_model('efficientnet_b4', pretrained=False, in_chans=3)
        self.base.classifier = torch.nn.Linear(self.base.classifier.in_features, num_classes)
    def forward(self, x):
        return self.base(x)

model = EfficientNetAudio(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()


N = 10  # сколько файлов показать
true_count=0
for i, row in df.iterrows():
    if i >= N:
        break
    fname = row['fname']
    true_label = row['label']
    filepath = f"{AUDIO_DIR}/{fname}"
    y = load_audio(filepath)
    y = process_length(y)
    mel_spec = create_mel_spectrogram(y)
    mel_spec = torch.tensor(mel_spec).float().unsqueeze(0).repeat(3, 1, 1)  
    X = mel_spec.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(X)
        pred = torch.argmax(logits, 1).item()
    pred_label = label_encoder.inverse_transform([pred])[0]
    if pred_label == true_label:
        true_count+=1
    print(f"Sample {i+1}: True: {true_label} | Predicted: {pred_label} | File: {fname}")
    
print (true_count)