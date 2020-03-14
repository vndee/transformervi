import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalization(nn.Module):
    def __init__(self, dimension, epsilon=1e-6):
        """
        Normalization
        :param dimension:
        :param epsilon:
        """
        super(Normalization, self).__init__()
        self.dimension = dimension
        self.alpha = nn.Parameter(torch.ones(self.dimension))
        self.bias = nn.Parameter(torch.zeros(self.dimension))
        self.epsilon = epsilon

    def forward(self, x):
        return self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(self, dimension, d_ff=2048, dropout_ratio=0.1):
        """
        Feedforward network
        :param dimension:
        :param d_ff:
        :param dropout_ratio:
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dimension, d_ff)
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear2 = nn.Linear(d_ff, dimension)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        return self.linear2(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, dimension, dropout_ratio=0.1):
        """
        Multi-head attention
        :param heads:
        :param dimension:
        :param dropout_ratio:
        """
        super(MultiHeadAttention, self).__init__()

        self.dimension = dimension
        self.d_k = self.dimension // heads
        self.heads = heads

        self.query = nn.Linear(self.dimension, self.dimension)
        self.key = nn.Linear(self.dimension, self.dimension)
        self.value = nn.Linear(self.dimension, self.dimension)

        self.dropout = nn.Dropout(dropout_ratio)
        self.out = nn.Linear(self.dimension, self.dimension)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        k = self.key(k).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        q = self.query(q).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)

        scores = MultiHeadAttention.self_attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.dimension)

        return self.out(concat)

    @staticmethod
    def self_attention(q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)

        return torch.matmul(scores, v)
