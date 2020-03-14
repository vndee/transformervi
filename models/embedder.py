import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class Embedder(nn.Module):
    def __init__(self, vocab_size, dimension):
        """
        Word embedding
        :param vocab_size:
        :param dimension:
        """
        super(Embedder, self).__init__()
        self.dimension = dimension
        self.embedding = nn.Embedding(vocab_size, dimension)

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoder(nn.Module):
    def __init__(self, dimension, max_sequence_length=200, dropout_ratio=0.1):
        """
        Positional encoding
        :param dimension:
        :param max_sequence_length:
        :param dropout_ratio:
        """
        super(PositionalEncoder, self).__init__()
        self.dimension = dimension
        self.dropout = nn.Dropout(dropout_ratio)

        mat = torch.zeros(max_sequence_length, self.dimension)
        for pos in range(max_sequence_length):
            for i in range(0, dimension, 2):
                mat[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.dimension)))
                mat[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.dimension)))

        mat = mat.unsqueeze(0)
        self.register_buffer('mat', mat)

    def forward(self, x):
        x = x * math.sqrt(self.dimension)
        positional_encoding = Variable(self.mat[:, :x.size(1)], requires_grad=False)
        if x.is_cuda():
            positional_encoding.cuda()

        return self.dropout(x + positional_encoding)
