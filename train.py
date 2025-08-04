# Use Modal to run serverless GPU for training
import sys
import pandas as pd
import torchaudio
import modal
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from model import AudioCNN
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

app = modal.App("sonimir")

image = (modal.Image.debian_slim(
).pip_install_from_requirements("requirements.txt").apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"]).run_commands([
    "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
    "cd /tmp && unzip esc50.zip",
    "mkdir -p /opt/esc50-data",
    "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
    "rm -rf /tmp/esc50.zip /tmp/ESC-50-master",
]).add_local_python_source("model"))

volume = modal.Volume.from_name(
    # this is our dataset
    "esc50-data", create_if_missing=True)
modal_volume = modal.Volume.from_name(
    "esc-model", create_if_missing=True)  # this is our trained model


class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split == "train":
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]

        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.metadata['label'] = self.metadata['category'].map(
            self.class_to_idx  # map the audio to an idx
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / \
            row['filename']  # datadir/audio/2-101616.wav

        waveform, sample_rate = torchaudio.load(audio_path)
        # waveform will look like [channels, samples] = [2, 44000]

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        # map the spectrogram to the classification label
        return spectrogram, row["label"]

# combine two audio clips to train model to deal with non-ideal audio data


def mixup_data(x, y):
    lam = np.random.beta(0.2, 0.2)

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # (0.7 * audio1) + (0.3 * audio2)
    mixed_x = lam * x + (1 - lam) * x[index, :]

    label_a, label_b = y, y[index]

    return mixed_x, label_a, label_b, lam

# lam = blend percentage
# find loss of blended audio using weighted calculation


def mixup_criterion(criterion, prediction, label_a, label_b, lam):
    return lam * criterion(prediction, label_a) + (1-lam) * criterion(prediction, label_b)


@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": modal_volume}, timeout=60 * 60 * 3)
# FIX /data file path above. The data is actually backed into the Docker container using the image code at top of this file
def train():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'/models/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)

    print("training")
    esc50_dir = Path("/opt/esc50-data")
    train_transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=22050, n_fft=1024,
                         hop_length=512, n_mels=128, f_min=0, f_max=11025),
        T.AmplitudeToDB(),
        # help prevent overfitting (dropout for audio)
        T.FrequencyMasking(freq_mask_param=30),
        # help prevent overfitting (dropout for audio)
        T.TimeMasking(time_mask_param=80),
    )

    val_transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=22050, n_fft=1024,
                         hop_length=512, n_mels=128, f_min=0, f_max=11025),
        T.AmplitudeToDB(),
    )

    train_dataset = ESC50Dataset(data_dir=esc50_dir, metadata_file=esc50_dir /
                                 "meta" / "esc50.csv", split="train", transform=train_transform)

    test_dataset = ESC50Dataset(data_dir=esc50_dir, metadata_file=esc50_dir /
                                "meta" / "esc50.csv", split="val", transform=val_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Val samples: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    scheduler = OneCycleLR(optimizer, max_lr=0.002, epochs=num_epochs,
                           steps_per_epoch=len(train_dataloader), pct_start=0.1)

    best_accuracy = 0.0
    print("Starting training")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for data, target in progress_bar:
            # spectrograms, labels
            data, target = data.to(device), target.to(device)

            # 30% of testing, use audio mixer
            if np.random.random() > 0.7:
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(
                    criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar(
            'Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Validation afer each epoch
        model.eval()

        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(
                    device)  # transfer to GPU if available
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)  # count how many samples
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(test_dataloader)

        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar(
            'Accuracy/Validation', accuracy, epoch)

        print(
            f"Epoch {epoch + 1} Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes
            }, '/models/best_model.pth')
            print(f"New best model saved w/ accuracy: {best_accuracy:.2f}%")

    writer.close()
    print(f'Training completed! Best accuracy: {best_accuracy:.2f}%')


@app.local_entrypoint()
def main():
    train.remote()
