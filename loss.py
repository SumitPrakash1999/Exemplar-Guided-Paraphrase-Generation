import torch
import torch.nn.functional as F
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
min_num = 1e-6

# Small constant to avoid log of zero
MIN_NUM = 1e-8

def compute_nll_loss(fc_out, fc_label, weight=None, label=None, reduction='mean', ope='log_softmax'):
    """
    Computes the Negative Log-Likelihood (NLL) loss.

    Args:
        fc_out (torch.Tensor): The predicted logits, shape [batch_size, vocab_size] or [batch_size, seq_len, vocab_size].
        fc_label (torch.Tensor): The ground truth labels, shape [batch_size] or [batch_size, seq_len].
        weight (torch.Tensor, optional): The weight for each element, shape [batch_size] or [batch_size, seq_len].
        label (torch.Tensor, optional): A mask to indicate valid labels, shape [batch_size] or [batch_size, seq_len].
        reduction (str, optional): Specifies the reduction to apply to the output. Options are 'none', 'mean'. Default is 'mean'.
        ope (str, optional): Specifies the operation to apply on the logits. Options are 'log_softmax' or 'log'. Default is 'log_softmax'.

    Returns:
        torch.Tensor: The computed NLL loss. If `reduction='mean'`, returns a scalar. Otherwise, returns a tensor of shape [batch_size].

    """
    # Apply the log operation as specified
    if ope == 'log':
        fc_out = (fc_out + MIN_NUM).log()
    else:
        fc_out = F.log_softmax(fc_out, dim=-1)

    # If label is not provided, assume a mask based on fc_label values
    if label is None:
        label = (fc_label > 0).float()

    # If weight is provided, apply it to the label
    if weight is not None:
        label *= weight

    # If the labels are of shape [batch_size], compute the loss per sample
    if len(fc_label.size()) == 1:
        return label * F.nll_loss(input=fc_out, target=fc_label, reduction=reduction)
    
    # If the labels are of shape [batch_size, seq_len], compute the loss for each sequence element
    elif len(fc_label.size()) == 2:
        # Compute the NLL loss per token, then apply the label mask
        loss = (label * F.nll_loss(input=fc_out.transpose(1, 2), target=fc_label, reduction='none')).sum(dim=-1) / (label.sum(dim=-1) + MIN_NUM)

        if reduction == 'mean':
            return loss.mean()
        else:
            return loss


def compute_entropy(prob):
    """
    Computes the entropy of a probability distribution.

    Args:
        prob (torch.Tensor): The probability distribution, shape [batch_size, num_classes].

    Returns:
        torch.Tensor: The entropy values for each batch, shape [batch_size].
    """
    return - (prob * (prob + MIN_NUM).log()).sum(dim=-1)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon) as described in:
    https://arxiv.org/pdf/2004.11362.pdf.
    This loss function also supports the unsupervised contrastive loss in SimCLR:
    https://arxiv.org/pdf/2002.05709.pdf.

    Args:
        temperature (float): The temperature scaling factor used for the contrastive loss.
        contrast_mode (str): Determines how to select the anchor features.
            - 'one': Use only the first view as the anchor.
            - 'all': Use all views as anchors.
        base_temperature (float): The base temperature for loss scaling.
    """
    def __init__(self, temperature=0.5, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Compute the contrastive loss.

        Args:
            features (torch.Tensor): The input features, shape [batch_size, n_views, feature_dim].
            labels (torch.Tensor, optional): Ground truth labels, shape [batch_size]. 
                If None, unsupervised SimCLR loss is computed.
            mask (torch.Tensor, optional): A custom contrastive mask, shape [batch_size, batch_size].
                Defines which samples belong to the same class. It is asymmetric.
        
        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        device = features.device

        # Ensure features has at least 3 dimensions (batch_size, n_views, feature_dim)
        if len(features.shape) < 3:
            raise ValueError('`features` must have at least 3 dimensions: [batch_size, n_views, feature_dim].')

        # Flatten the features to shape [batch_size * n_views, feature_dim] if there are more than 3 dimensions
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # Handle label and mask logic
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`.')
        
        if labels is None and mask is None:
            # Default to identity mask (same class for all samples)
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('The number of labels must match the number of features.')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)

        # Select anchor features based on contrast mode
        if self.contrast_mode == 'one':
            anchor_features = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_features = contrast_features
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast mode: {self.contrast_mode}")

        # Compute contrastive logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_features, contrast_features.T),
            self.temperature
        )

        # For numerical stability, subtract the max value from the logits
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile the mask for all anchor and contrast pairs
        mask = mask.repeat(anchor_count, contrast_count)

        # Mask out self-contrast cases (anchor against itself)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute the log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute the mean log-probability over positive samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Final loss computation
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss