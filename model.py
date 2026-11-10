"""The project's single machine learning algorithm: a character 1D CNN."""
import torch
from torch import nn
import config


def model_settings():
    return dict(vocabulary_size=config.VOCABULARY_SIZE,
                embedding_dim=config.EMBEDDING_DIM, num_filters=config.NUM_FILTERS,
                kernel_size=config.KERNEL_SIZE, stride=config.STRIDE,
                padding=config.PADDING, dropout_rate=config.DROPOUT_RATE)


class PathCNN(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, num_filters, kernel_size,
                 stride, padding, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(embedding_dim, num_filters, kernel_size, stride, padding)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(num_filters, 1)

    def forward(self, ids):
        # [batch, length] -> [batch, length, embedding_dim]: learned character vectors.
        vectors = self.embedding(ids).transpose(1, 2)
        # [batch, embedding_dim, length] -> [batch, filters, windows].
        # Conv1D computes learned weighted sums of neighboring character vectors + bias.
        features = torch.relu(self.conv(vectors))  # ReLU = max(0, x), a nonlinear activation.
        pooled = features.amax(dim=2)  # [batch, filters]: strongest match anywhere per filter.
        return self.output(self.dropout(pooled)).squeeze(1)  # [batch], one raw logit per path.
