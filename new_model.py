import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from timm import create_model
import matplotlib.pyplot as plt
import librosa
import seaborn as sns

#конфигурация

AUDIO_DIR = "audio_train/train"
CSV_PATH = "train.csv"
SAVE_DIR = "res/results"
SAMPLE_RATE = 32000
N_MELS = 128
HOP_LENGTH = 512
TARGET_FRAMES = 380
BATCH_SIZE = 32
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(SAVE_DIR, exist_ok=True)

# тут обработка
def load_audio(filepath, sr=SAMPLE_RATE):
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
# создание спектра
def create_mel_spectrogram(y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    # Привести к нужному числу фреймов
    if mel_db.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :TARGET_FRAMES]
    return mel_db

#Аугментации для аудио
def Agments_aud(y):
    # Time shift
    shift = int(random.uniform(-0.1, 0.1) * len(y))
    y = np.roll(y, shift)
    # White noise
    noise_level = random.uniform(0.001, 0.01)
    noise = np.random.randn(len(y)) * noise_level
    y = y + noise
    return y

# Функция для равенства элементов класса классов
def balance_data(df):
    max_samples = df['label'].value_counts().max()
    dfs = []
    for label in df['label'].unique():
        class_df = df[df['label'] == label]
        balanced = resample(class_df, replace=True, n_samples=max_samples, random_state=SEED)
        dfs.append(balanced)
    balanced_df = pd.concat(dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return balanced_df


def create_split_data(df, test_size=0.2):
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=SEED
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class AudioDataset(Dataset):
    def __init__(self, df, audio_dir, label_encoder, augment=False):
        self.df = df
        self.audio_dir = audio_dir
        self.label_encoder = label_encoder
        self.augment = augment
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['fname']
        label = row['label']
        label_encoded = self.label_encoder.transform([label])[0]
        filepath = os.path.join(self.audio_dir, fname)
        y = load_audio(filepath)
        if self.augment:
            y = Agments_aud(y)
        y = process_length(y)
        mel_spec = create_mel_spectrogram(y)
        mel_spec = torch.tensor(mel_spec).float()
        mel_spec = mel_spec.unsqueeze(0).repeat(3, 1, 1)  
        return mel_spec, label_encoded


df = pd.read_csv(CSV_PATH)
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])
NUM_CLASSES = len(label_encoder.classes_)

# Балансировка
balanced_df = balance_data(df)
# Сплит
train_df, val_df = create_split_data(balanced_df, test_size=0.2)
# Датасеты и DataLoader'ы
train_dataset = AudioDataset(train_df, AUDIO_DIR, label_encoder, augment=True)
val_dataset = AudioDataset(val_df, AUDIO_DIR, label_encoder, augment=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

#модель

class EfficientNetAudio(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = create_model('efficientnet_b4', pretrained=True, in_chans=3)
        self.base.classifier = nn.Linear(self.base.classifier.in_features, num_classes)
    def forward(self, x):
        return self.base(x)


def train_model(
    model, train_loader, val_loader, num_epochs=70, max_lr=1e-3, device='cuda',
    early_stop_f1=0.87
):
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=max_lr)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr,
        steps_per_epoch=steps_per_epoch, epochs=num_epochs
    )
    best_f1 = 0
    best_model_wts = None
    best_val_labels = []
    best_val_preds = []
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        #Тренировка
        model.train()
        train_loss, train_preds, train_labels = 0, [], []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * X.size(0)
            train_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            train_labels.extend(y.cpu().numpy())
        train_loss /= len(train_loader.dataset)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        print(f"Train loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        #Валидация
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = loss_fn(logits, y)
                val_loss += loss.item() * X.size(0)
                val_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        print(f"Val loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        #Выбор лучшей модели и сохранене
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = model.state_dict()
            best_val_labels = val_labels.copy()
            best_val_preds = val_preds.copy()
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_efficientnet_b4.pt"))
            print(f"Model saved (f1={best_f1:.3f})")
        if best_f1 >= early_stop_f1:
            print('Target F1 reached, stopping training!')
            break
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['train_f1'], label='Train F1')
        plt.plot(history['val_f1'], label='Val F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1')
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    #Сохранение истории
    torch.save(history, os.path.join(SAVE_DIR, "train_history.pt"))
    #ГРафики
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', marker='o')
    plt.plot(history['val_loss'], label='Validation', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Train', marker='o')
    plt.plot(history['val_f1'], label='Validation', marker='o')
    plt.title('Training and Validation F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "train_val_curves.png"))
    plt.show()
    #Матрица ошибок лучшей модели
    print("\nConfusion Matrix for the best model:")
    cm = confusion_matrix(best_val_labels, best_val_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (best model)')
    plt.tight_layout()
    cm_path = os.path.join(SAVE_DIR, "confusion_matrix_best_model.png")
    plt.savefig(cm_path)
    plt.show()
    print(f"Confusion matrix saved to: {cm_path}")
    print(classification_report(best_val_labels, best_val_preds, target_names=label_encoder.classes_))
    return model

model = EfficientNetAudio(num_classes=NUM_CLASSES)
trained_model = train_model(
    model, train_loader, val_loader, num_epochs=70, max_lr=1e-3, device=DEVICE
)

