import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
# from cnn_ import MelToZEncoder, ZNet

from torch.profiler import profile, record_function, ProfilerActivity



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

        return z_mu, F.softplus(z_logvar) + 1e-8

def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, dim): Samples
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn(m.shape).to(device)
    z = z * torch.sqrt(v).to(device)
    z = z + m
    return z

def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution

    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance

    Returns:
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.z_dim = 128
        self.z_text_dim = 768
        self.mel2z = MelToZEncoder(preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                                    2*self.z_dim)
        self.znet = ZNet(2 * self.z_dim, self.z_text_dim, self.z_dim)
        
        self.encoder = Encoder(model_config, self.z_dim)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            2 * preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()
        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
        

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        bert_embeddings=None,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        train=False
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        print(train)

        #inference
        if not train:
            mu_enc, var_enc = None, None
            z = torch.randn(batch_size, z_dim)
        
        # training

        
        else:
            # print("Mel shape: ", mels.shape)
            # print("bert shape: ", bert_embeddings.shape)
            z_mel = self.mel2z(mels)
            print("Test z_mel", z_mel.isnan().any())
            z_pbert = bert_embeddings
            print("Test z_pbert", z_pbert.isnan().any())
            mu_enc, var_enc = self.znet(z_mel, z_pbert)

            print("Test mu_enc", mu_enc.isnan().any())
            print("Test var_enc", var_enc.isnan().any())

            # print("mu_enc, var_enc ",  mu_enc.shape, var_enc.shape)

            z = sample_gaussian(mu_enc, var_enc)
            print("Test var_enc", z.isnan().any())
            # print("z normal(mu_enc, var_enc) ",  z.shape)
        
        print("Test after VAE encoder", z.isnan().any())


        output = self.encoder(texts, src_masks, z)
        print('Output ', output.shape)
        print("Test after encoder", output.isnan().any())

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)

        print("Test after decoder", output.isnan().any())

        print("output after decoder", output.shape)

        output = self.mel_linear(output)

        print("output after mel_linear", output.shape)

        mu_dec, var_dec = gaussian_parameters(output)
        var_dec = F.softplus(var_dec) + 1e-8

        print("output after mu_dec, var_dec", mu_dec.shape, var_dec.shape)


        if not train:
            output = sample_gaussian(mu_dec, var_dec)
        else:
            output = None


        # postnet_output = output #self.postnet(output) + output
        return (
            mu_enc, 
            var_enc,
            mu_dec,
            var_dec,
            output,
            # postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )