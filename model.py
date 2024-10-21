import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from tqdm import tqdm
import numpy as np

def load_data(file_path='iemocap_data.pkl'):
    with open(file_path, 'rb') as f:
        emotion_data = pickle.load(f)

    mfcc_features = [item[0] for item in emotion_data]  # MFCC features
    emotion_labels = [item[2] for item in emotion_data]  # Emotion labels

    emotion_mapping = {'neu': 0, 'happy': 1, 'sad': 2, 'ang': 3, 'exc': 4, 'fru': 5, 'sur': 6}
    
    mfcc_features_filtered = []
    emotion_labels_filtered = []

    for mfcc, emotion in zip(mfcc_features, emotion_labels):
        if emotion in emotion_mapping:
            mfcc_features_filtered.append(mfcc)
            emotion_labels_filtered.append(emotion_mapping[emotion])

    mfcc_train, mfcc_test, emotion_train, emotion_test = train_test_split(
        mfcc_features_filtered, emotion_labels_filtered, test_size=0.2, random_state=42)

    return mfcc_train, mfcc_test, emotion_train, emotion_test

class IEMOCAPDataset(torch.utils.data.Dataset):
    def __init__(self, mfcc_features, emotion_labels):
        self.mfcc_features = mfcc_features
        self.emotion_labels = emotion_labels
    
    def __len__(self):
        return len(self.mfcc_features)
    
    def __getitem__(self, idx):
        mfcc_tensor = torch.tensor(self.mfcc_features[idx], dtype=torch.float32)
        label = torch.tensor(self.emotion_labels[idx], dtype=torch.long)
        return mfcc_tensor, label

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        conv_out = self.conv_layers(x)
        conv_out = torch.mean(conv_out, dim=2)
        frame_level_features = self.fc(conv_out)
        return frame_level_features

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim  

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim) 
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        mu_logvar = self.encoder(x).view(-1, 2, self.latent_dim)  
        mu = mu_logvar[:, 0, :]  
        logvar = mu_logvar[:, 1, :] 

        z = self.reparameterize(mu, logvar)

        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar


class PrecursorSpeakerLayer(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(PrecursorSpeakerLayer, self).__init__()
        self.vae = VAE(input_dim, latent_dim)
    
    def forward(self, x):
        reconstructed_x, mu, logvar = self.vae(x)
        return reconstructed_x, mu, logvar

class DisentangledContentLayer(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DisentangledContentLayer, self).__init__()
        self.vae = VAE(latent_dim, latent_dim)
        self.project_to_latent = nn.Linear(input_dim, latent_dim)
    
    def forward(self, x, speaker_mu):
        projected_x = self.project_to_latent(x)
        content_input = projected_x - speaker_mu
        reconstructed_x, content_mu, content_logvar = self.vae(content_input)
        return reconstructed_x, content_mu, content_logvar

class FinalSpeakerLayer(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(FinalSpeakerLayer, self).__init__()
        self.vae = VAE(latent_dim, latent_dim)
        self.project_to_latent = nn.Linear(input_dim, latent_dim)
    
    def forward(self, x, content_mu):
        projected_x = self.project_to_latent(x)
        speaker_input = projected_x - content_mu
        reconstructed_x, final_speaker_mu, final_speaker_logvar = self.vae(speaker_input)
        return reconstructed_x, final_speaker_mu, final_speaker_logvar

class TemporalAggregation(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(TemporalAggregation, self).__init__()
        self.layer1 = PrecursorSpeakerLayer(input_dim, latent_dim)
        self.layer2 = DisentangledContentLayer(input_dim, latent_dim)
        self.layer3 = FinalSpeakerLayer(input_dim, latent_dim)
    
    def forward(self, x):
        precursor_output, speaker_mu, speaker_logvar = self.layer1(x)
        content_output, content_mu, content_logvar = self.layer2(x, speaker_mu)
        final_speaker_output, final_speaker_mu, final_speaker_logvar = self.layer3(x, content_mu)
        return precursor_output, final_speaker_mu, final_speaker_logvar

class EmotionRecognitionBranch(nn.Module):
    def __init__(self, input_dim, num_emotions):
        super(EmotionRecognitionBranch, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_emotions)
        )
    
    def forward(self, x):
        emotion_logits = self.fc_layers(x)
        return emotion_logits

class EmotionAdversarialDiscriminator(nn.Module):
    def __init__(self, latent_dim, num_emotions):
        super(EmotionAdversarialDiscriminator, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_emotions)
        )
    
    def forward(self, speaker_embedding):
        emotion_logits = self.fc_layers(speaker_embedding)
        return emotion_logits

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, speaker_embedding):
        reconstructed_input = self.decoder(speaker_embedding)
        return reconstructed_input

class FullModelWithEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_emotions):
        super(FullModelWithEncoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.temporal_aggregation = TemporalAggregation(128, latent_dim)
        self.emotion_recognition = EmotionRecognitionBranch(128, num_emotions)
        self.emotion_discriminator = EmotionAdversarialDiscriminator(latent_dim, num_emotions)
        self.decoder = Decoder(latent_dim, input_dim)
    
    def forward(self, x):
        encoded_features = self.encoder(x)
        precursor_output, final_speaker_mu, final_speaker_logvar = self.temporal_aggregation(encoded_features)
        emotion_logits = self.emotion_recognition(encoded_features)
        adversarial_emotion_logits = self.emotion_discriminator(final_speaker_mu)
        reconstructed_input = self.decoder(final_speaker_mu)
        return emotion_logits, adversarial_emotion_logits, reconstructed_input

def build_model(input_dim=13, latent_dim=16, hidden_dim=64, num_emotions=6):
    return FullModelWithEncoder(input_dim, latent_dim, hidden_dim, num_emotions)

def calculate_class_weights(labels):
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)

def train_model(model, train_loader, test_loader, class_weights, num_epochs=100, lr=0.0001, save_path='best_model.pth'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    emotion_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    reconstruction_loss_fn = nn.MSELoss()

    best_val_accuracy = 0.0
    early_stop_count = 0
    early_stop_patience = 10
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        model.train()
        total_emotion_loss = 0
        all_preds, all_labels = [], []

        for mfcc_batch, emotion_batch in train_loader:
            optimizer.zero_grad()
            emotion_logits, adversarial_emotion_logits, reconstructed_mfcc = model(mfcc_batch)
            reconstructed_mfcc = reconstructed_mfcc.unsqueeze(-1).expand_as(mfcc_batch)

            emotion_loss = emotion_loss_fn(emotion_logits, emotion_batch)
            reconstruction_loss = reconstruction_loss_fn(reconstructed_mfcc, mfcc_batch)
            adversarial_loss = -emotion_loss_fn(adversarial_emotion_logits, emotion_batch)

            total_loss = emotion_loss + reconstruction_loss + adversarial_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_emotion_loss += emotion_loss.item()
            preds = torch.argmax(emotion_logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(emotion_batch.cpu().numpy())

        train_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Train Emotion Loss: {total_emotion_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")

        model.eval()
        val_emotion_loss = 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for mfcc_batch, emotion_batch in test_loader:
                emotion_logits, adversarial_emotion_logits, reconstructed_mfcc = model(mfcc_batch)
                reconstructed_mfcc = reconstructed_mfcc.unsqueeze(-1).expand_as(mfcc_batch)

                emotion_loss = emotion_loss_fn(emotion_logits, emotion_batch)
                reconstruction_loss = reconstruction_loss_fn(reconstructed_mfcc, mfcc_batch)
                adversarial_loss = -emotion_loss_fn(adversarial_emotion_logits, emotion_batch)

                val_emotion_loss += emotion_loss.item()

                val_preds = torch.argmax(emotion_logits, dim=1).cpu().numpy()
                all_val_preds.extend(val_preds)
                all_val_labels.extend(emotion_batch.cpu().numpy())

        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        print(f"Val Emotion Loss: {val_emotion_loss / len(test_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with validation accuracy: {val_accuracy:.4f}")
            early_stop_count = 0
        else:
            early_stop_count += 1

        scheduler.step(val_emotion_loss / len(test_loader))

        if early_stop_count >= early_stop_patience:
            print("Early stopping triggered.")
            break

    print("Training finished!")

def pad_collate_fn(batch):
    mfccs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    max_len = max([mfcc.shape[1] for mfcc in mfccs])
    padded_mfccs = [torch.nn.functional.pad(mfcc, (0, max_len - mfcc.shape[1])) for mfcc in mfccs]
    mfcc_tensor = torch.stack(padded_mfccs)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return mfcc_tensor, label_tensor

mfcc_train, mfcc_test, emotion_train, emotion_test = load_data()
label_mapping = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
emotion_train = [label_mapping[label] for label in emotion_train]
emotion_test = [label_mapping[label] for label in emotion_test]

class_weights = calculate_class_weights(emotion_train).to('cpu')

train_dataset = IEMOCAPDataset(mfcc_train, emotion_train)
test_dataset = IEMOCAPDataset(mfcc_test, emotion_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

model = build_model(input_dim=13).to('cpu')
train_model(model, train_loader, test_loader, class_weights)
