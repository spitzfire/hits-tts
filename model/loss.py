import torch
import torch.nn as nn

def log_normal(x, m, v, mask):
    """
    Computes the element-wise log probability of a Gaussian, then applies a mask,
    and sums over the last dimension for the non-masked elements.

    Args:
        x: tensor: (batch, time_steps, dim): Observation
        m: tensor: (batch, time_steps, dim): Mean
        v: tensor: (batch, time_steps, dim): Variance
        mask: tensor: (batch, time_steps): Mask to apply

    Return:
        log_prob: tensor: (batch): Log probability of each sample in the batch.
    """
    dist = torch.distributions.Normal(m, torch.sqrt(v))
    log_prob = dist.log_prob(x)

    # Expand mask to match the shape of log_prob and apply it
    mask = mask.unsqueeze(-1).expand_as(log_prob)
    log_prob_masked = log_prob * mask

    # Sum over the last dimension
    return torch.sum(log_prob_masked, dim=-1)

def kl_normal(qm, qv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    # element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    # qv = F.softplus(qv) + 1e-8
    element_wise = 0.5 * (-torch.log(qv) + qv + qm.pow(2) - 1)
    # element_wise = 0.5 * (torch.log(qv) + qv + (qm).pow(2) - 1)
    kl = element_wise.sum(-1)
    return kl

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[7:13]
        (   
            mu_enc, 
            var_enc,
            mu_dec,
            var_dec,
            output,
            # mel_predictions,
            # postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        # print(mel_masks)
        # print(mel_masks.shape)

        # print(mu_dec.shape, var_dec.shape, mel_targets.shape)
        # mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        # mu_dec = mu_dec.masked_select(mel_masks.unsqueeze(-1))
        # var_dec = var_dec.masked_select(mel_masks.unsqueeze(-1))

        
        # postnet_mel_predictions = postnet_mel_predictions.masked_select(
        #     mel_masks.unsqueeze(-1)
        # )
        # mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        # mel_loss = self.mae_loss(mel_predictions, mel_targets)
        mel_loss = -1 * torch.mean(log_normal(mel_targets, mu_dec, var_dec, mel_masks))
        
        
        kl_loss = torch.mean(kl_normal(mu_enc, var_enc))

        # postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + kl_loss + duration_loss + pitch_loss + energy_loss
        )

        # total_loss = (
        #     mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        # )

        return (
            total_loss,
            mel_loss,
            kl_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
