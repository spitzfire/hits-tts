import os
import glob
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


class MelToZEncoder(nn.Module):
    def __init__(self, mel_channels, z_dim):
        """
        A convolutional network to encode variable-length mel spectrograms into a latent vector z.
        The input format is (batch_size, time_steps, mel_channels).
        
        Parameters:
        mel_channels (int): The number of channels in the input mel spectrogram.
        z_dim (int): The dimension of the latent vector z.
        """
        super(MelToZEncoder, self).__init__()

        # Convolutional layers operating on the mel_channels dimension
        self.conv1 = nn.Conv1d(mel_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)

        # Adaptive average pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers for mean and log variance of z
        self.fc = nn.Linear(512, z_dim)
        # self.fc_logvar = nn.Linear(512, z_dim)

    def forward(self, mel_spectrogram):
        # Reshape the input to fit the convolutional layer (batch_size, mel_channels, time_steps)
        x = mel_spectrogram.permute(0, 2, 1)

        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Adaptive pooling to handle variable length inputs
        x = self.adaptive_pool(x).squeeze(-1)

        # Calculate mean and log variance of z
        z_mu = self.fc(x)
        return z_mu
        # z_logvar = self.fc_logvar(x)

        # return z_mu, z_logvar

class ZNet(nn.Module):
    def __init__(self, z_mel_dim, z_text_dim, z_dim):
        """
        A neural network to fuse z_mel and z_text into a single latent variable z.

        Parameters:
        z_mel_dim (int): Dimension of the mel latent variable z_mel.
        z_text_dim (int): Dimension of the text latent variable z_text.
        z_dim (int): Dimension of the fused latent variable z.
        """
        super(ZNet, self).__init__()

        # Fully connected layers to process z_mel and z_text
        self.fc_mel = nn.Linear(z_mel_dim, z_dim)
        self.fc_text = nn.Linear(z_text_dim, z_dim)

        # Fully connected layer to combine the processed z_mel and z_text
        self.fc_combine = nn.Linear(2 * z_dim, z_dim)

        # Fully connected layers for mean and log variance of the fused z
        self.fc_mu = nn.Linear(z_dim, z_dim)
        self.fc_logvar = nn.Linear(z_dim, z_dim)

    def forward(self, z_mel, z_text):
        """
        Forward pass of the ZFusionNet.

        Parameters:
        z_mel (Tensor): The mel latent variable of shape (batch_size, z_mel_dim).
        z_text (Tensor): The text latent variable of shape (batch_size, z_text_dim).

        Returns:
        Tuple[Tensor, Tensor]: The mean and log variance of the fused latent variable z.
        """
        # Process z_mel and z_text
        z_mel_processed = F.relu(self.fc_mel(z_mel))
        z_text_processed = F.relu(self.fc_text(z_text))

        # Concatenate and combine the processed latent variables
        z_combined = torch.cat((z_mel_processed, z_text_processed), dim=1)
        z_combined = F.relu(self.fc_combine(z_combined))

        # Calculate mean and log variance of the fused z
        z_mu = self.fc_mu(z_combined)
        z_logvar = self.fc_logvar(z_combined)

        return z_mu, z_logvar

# # Example usage
# z_mel_dim = 256  # Dimension of the mel latent variable
# z_text_dim = 768  # Dimension of the text latent variable
# z_dim = 128  # Dimension of the fused latent variable
# model = ZFusionNet(z_mel_dim, z_text_dim, z_dim)

# # Example inputs
# z_mel_sample = torch.randn(batch_size, z_mel_dim)  # Random example for z_mel
# z_text_sample = torch.randn(batch_size, z_text_dim)  # Random example for z_text

# # Get the fused latent variable
# z_mu, z_logvar = model(z_mel_sample, z_text_sample)
# z_mu.shape, z_logvar.shape  # Shapes of the mean and log variance of z



# # Example usage
# mel_channels = 80  # Typical number of mel spectrogram channels
# z_dim = 128  # Dimension of the latent vector z
# model = MelToZEncoder(mel_channels, z_dim)

# # Example mel spectrogram tensor (batch_size, mel_channels, time_steps)
# example_mel_spectrogram = torch.unsqueeze(torch.rand(200, mel_channels), 0)  # Random example
# print(example_mel_spectrogram.shape)
# z_mu, z_logvar = model(example_mel_spectrogram)
# print(z_mu.shape, z_logvar.shape)
# # # Base directory where the 'mel' folders are located
# base_dir = '/home/disk_2/suvrat2/ps/FastSpeech2/preprocessed_data/LibriTTS'

# # # Directory containing the mel spectrograms
# mel_dir = os.path.join(base_dir, 'mel')

# # # Iterate through all .npy files in the mel directory
# for mel_file in glob.glob(os.path.join(mel_dir, '*.npy')):
#     # Load the mel spectrogram file
#     mel =  torch.unsqueeze(torch.from_numpy(np.load(mel_file)), 0) 
#     print(mel.shape)
#     z_mu, z_logvar = model(mel)
#     print(z_mu.shape, z_logvar.shape)  # Shapes of the mean and log variance of z
