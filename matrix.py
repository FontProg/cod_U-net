import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from timm import create_model
import os
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

SAVE_DIR = "results_new_model"
AUDIO_DIR = "new_test_model/data_new_model/val_split"
CSV_PATH = "new_test_model/data_new_model/val_split/val_annotations.csv"
MODEL_PATH = os.path.join(SAVE_DIR, "best_efficientnet_b4_f1_0.927.pt") 
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_MELS = 128
HOP_LENGTH = 512
TARGET_FRAMES = 380


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


class AudioDataset(Dataset):
    def __init__(self, df, audio_dir, label_encoder):
        self.df = df
        self.audio_dir = audio_dir
        self.label_encoder = label_encoder
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['fname']
        label = row['label']
        label_encoded = self.label_encoder.transform([label])[0]
        filepath = os.path.join(self.audio_dir, fname)
        y = load_audio(filepath)
        y = process_length(y)
        mel_spec = create_mel_spectrogram(y)
        mel_spec = torch.tensor(mel_spec).float()
        mel_spec = mel_spec.unsqueeze(0).repeat(3, 1, 1)
        return mel_spec, label_encoded


df = pd.read_csv(CSV_PATH)
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])
NUM_CLASSES = len(label_encoder.classes_)
val_dataset = AudioDataset(df, AUDIO_DIR, label_encoder)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

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

all_preds = []
all_labels = []
with torch.no_grad():
    for X, y in val_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        preds = torch.argmax(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))