import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import shutil

# Пути к исходным данным
CSV_PATH = "train.csv"  # ваш исходный csv с fname,label
AUDIO_DIR = "audio_train/train"  # папка с .wav файлами

# Пути для сохранения новых данных
TRAIN_DIR = "new_test_model/data_new_model/train_split"
VAL_DIR = "new_test_model/data_new_model/val_split"
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Чтение исходных аннотаций
df = pd.read_csv(CSV_PATH)

# Стратифицированное разбиение
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
for train_idx, val_idx in splitter.split(df['fname'], df['label']):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

# Копирование файлов (опционально, если нужно разнести файлы по папкам)
for _, row in train_df.iterrows():
    src = os.path.join(AUDIO_DIR, row['fname'])
    dst = os.path.join(TRAIN_DIR, row['fname'])
    shutil.copy2(src, dst)
for _, row in val_df.iterrows():
    src = os.path.join(AUDIO_DIR, row['fname'])
    dst = os.path.join(VAL_DIR, row['fname'])
    shutil.copy2(src, dst)

# Сохранение аннотаций
train_df.to_csv("train_annotations.csv", index=False)
val_df.to_csv("val_annotations.csv", index=False)

print(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples")
print("Классы в train:", train_df['label'].nunique())
print("Классы в val:", val_df['label'].nunique())