"""The project's single machine learning algorithm: a character 1D CNN."""
import torch
from torch import nn
import config
from preprocessing import NUM_METADATA_FEATURES


def model_settings():
    return dict(vocabulary_size=config.VOCABULARY_SIZE,
                embedding_dim=config.EMBEDDING_DIM, num_filters=config.NUM_FILTERS,
                kernel_size=config.KERNEL_SIZE, stride=config.STRIDE,
                padding=config.PADDING, dropout_rate=config.DROPOUT_RATE,
                num_extra_features=NUM_METADATA_FEATURES if config.USE_METADATA_FEATURES else 0)


class PathCNN(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, num_filters, kernel_size,
                 stride, padding, dropout_rate, num_extra_features=0):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(embedding_dim, num_filters, kernel_size, stride, padding)
        self.dropout = nn.Dropout(dropout_rate)
        # Optional metadata (size, depth, file-type group) joins the pooled features; 0 = paths only.
        self.num_extra_features = num_extra_features
        self.output = nn.Linear(num_filters + num_extra_features, 1)

    def forward(self, ids, extra=None):
        # [batch, length] -> [batch, length, embedding_dim]: learned character vectors.
        vectors = self.embedding(ids).transpose(1, 2)
        # [batch, embedding_dim, length] -> [batch, filters, windows].
        # Conv1D computes learned weighted sums of neighboring character vectors + bias.
        features = torch.relu(self.conv(vectors))  # ReLU = max(0, x), a nonlinear activation.
        pooled = features.amax(dim=2)  # [batch, filters]: strongest match anywhere per filter.
        pooled = self.dropout(pooled)
        if self.num_extra_features:
            if extra is None:
                extra = pooled.new_zeros(pooled.shape[0], self.num_extra_features)
            pooled = torch.cat([pooled, extra], dim=1)  # [batch, filters + extra]
        return self.output(pooled).squeeze(1)  # [batch], one raw logit per path.
