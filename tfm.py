import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        print(self.pe[][0])
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
# class PositionalEncoding(nn.Module):
#     def __init__(self, embedding_size, max_seq_len=200):
#         super(PositionalEncoding, self).__init__()
#         self.embedding_size = embedding_size
#         self.max_seq_len = max_seq_len
#         self.pe = self.get_positional_encoding(max_seq_len, embedding_size)
        
#     def get_positional_encoding(self, seq_len, embedding_size):
#         pe = torch.zeros(seq_len, embedding_size)
#         position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         return pe.unsqueeze(0)
        
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :]
#         return x

class TFM(nn.Module):
    def __init__(self, input_size=2054, output_size=6, num_layers=4, num_heads=2, hidden_size=512, dropout=0.2, mask_prob=0.15):
        super(TFM, self).__init__()
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout),
            num_layers
        )
        self.decoder = nn.Linear(input_size, output_size)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


src = torch.rand(24, 32, 518)
model = TFM(input_size=518)
# model = PositionalEncoding(d_model=518)
out = model(src)
out = out.permute(1,0,2)
print(out.shape)