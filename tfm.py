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
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TFM(nn.Module):
    def __init__(self, input_size=2054, output_size=6, num_layers=4, num_heads=2, hidden_size=512, dropout=0.2, mask_prob=0.15):
        super(TFM, self).__init__()
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout),
            num_layers
        )
        self.decoder = nn.Linear(input_size, output_size)
        self.mask_prob = mask_prob
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):
        x = self.pos_encoder(x)
        
        # Calculate the number of elements to be masked in the input tensor
        mask_size = int(math.ceil(self.mask_prob * x.size(0) * x.size(1)))
        print(mask_size)
        # Generate a binary mask with probability of each element to be masked
        mask = (torch.rand(x.size()) < self.mask_prob).float()
        
        print(mask.shape)
        # Set the mask to 0 for positions that do not need to be masked
        mask[:mask_size] = 0
        
        # Apply the mask to the input tensor
        x = x * mask

        x = self.encoder(x)
        x = self.decoder(x)
        return x


src = torch.rand(24, 32, 518)
model = TFM(input_size=518)
# model = PositionalEncoding(d_model=518)
out = model(src)
out = out.permute(1,0,2)
print(out.shape)