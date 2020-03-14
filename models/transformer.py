import torch
import torch.nn as nn
from models.utils import clone_module
from models.embedder import Embedder, PositionalEncoder
from models.layer import MultiHeadAttention, FeedForward, Normalization


class EncoderLayer(nn.Module):
    def __init__(self, dimension, heads, dropout_ratio=0.1):
        """
        Transformer encoder layer
        :param dimension:
        :param heads:
        :param dropout_ratio:
        """
        super(EncoderLayer, self).__init__()
        self.dimension = dimension
        self.norm_1 = Normalization(self.dimension)
        self.norm_2 = Normalization(self.dimension)
        self.multihead_attention = MultiHeadAttention(heads, self.dimension, dropout_ratio=dropout_ratio)
        self.feedforward = FeedForward(self.dimension, dropout_ratio=dropout_ratio)
        self.dropout_1 = nn.Dropout(dropout_ratio)
        self.dropout_2 = nn.Dropout(dropout_ratio)

    def forward(self, x, mask):
        y = self.norm_1(x)
        x = x + self.dropout_1(self.multihead_attention(y, y, y, mask))
        y = self.norm_2(x)
        x = x + self.dropout_2(self.feedforward(y))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dimension, heads, dropout_ratio=0.1):
        """
        Transformer decoder layer
        :param dimension:
        :param heads:
        :param dropout_ratio:
        """
        super(DecoderLayer, self).__init__()
        self.dimension = dimension

        self.norm_1 = Normalization(self.dimension)
        self.norm_2 = Normalization(self.dimension)
        self.norm_3 = Normalization(self.dimension)

        self.dropout_1 = nn.Dropout(dropout_ratio)
        self.dropout_2 = nn.Dropout(dropout_ratio)
        self.dropout_3 = nn.Dropout(dropout_ratio)

        self.multihead_attention_1 = MultiHeadAttention(heads, self.dimension, dropout_ratio=dropout_ratio)
        self.multihead_attention_2 = MultiHeadAttention(heads, self.dimension, dropout_ratio=dropout_ratio)
        self.feedforward = FeedForward(self.dimension, dropout_ratio=dropout_ratio)

    def forward(self, x, e_outputs, src_mask, tgt_mask):
        y = self.norm_1(x)
        x = x + self.dropout_1(self.multihead_attention_1(y, y, y, tgt_mask))

        y = self.nomr_2(x)
        x = x + self.dropout_2(self.multihead_attention_2(y, e_outputs, e_outputs, src_mask))

        y = self.norm_3(x)
        x = x + self.dropout_3(self.feedforward(y))

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, dimension, N, heads, dropout_ratio):
        """
        Transformer Encoder
        :param vocab_size:
        :param dimension:
        :param N:
        :param heads:
        :param dropout_ratio:
        """
        super(Encoder, self).__init__()
        self.N = N
        self.embedding = Embedder(vocab_size, dimension)
        self.positional_encoding = PositionalEncoder(dimension, dropout_ratio)
        self.layers = clone_module(EncoderLayer(dimension, heads, dropout_ratio), N)
        self.normalization = Normalization(dimension)

    def forward(self, src, mask):
        x = self.embedding(src)
        x = self.positional_encoding(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)

        return self.normalization(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, dimension, N, heads, dropout_ratio):
        """
        Transformer Decoder
        :param vocab_size:
        :param dimension:
        :param N:
        :param heads:
        :param dropout:
        """
        super(Decoder, self).__init__()
        self.N = N
        self.embedding = Embedder(vocab_size, dimension)
        self.positional_encoding = PositionalEncoder(dimension, dropout_ratio)
        self.layers = clone_module(DecoderLayer(dimension, heads, dropout_ratio), N)
        self.normalization = Normalization(dimension)

    def forward(self, tgt, e_ouputs, src_mask, tgt_mask):
        x = self.embedding(tgt)
        x = self.positional_encoding(x)
        for i in range(self.N):
            x = self.layers[i](x, e_ouputs, src_mask, tgt_mask)

        return self.normalization(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, dimension, N, heads, dropout_ratio):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab, dimension, N, heads, dropout_ratio)
        self.decoder = Decoder(tgt_vocab, dimension, N, heads, dropout_ratio)
        self.out = nn.Linear(dimension, tgt_vocab)

    def forward(self, src, tgt, src_mask, tgt_mask):
        e_outputs = self.encoder(src, src_mask)
        d_outputs = self.decoder(tgt, e_outputs, src_mask, tgt_mask)
        return self.output(d_outputs)

