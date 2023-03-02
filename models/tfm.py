import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=1, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.positional_encoding = nn.Embedding(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x.shape = (batch_size, 24, 1027)
        batch_size = x.shape[0]

        # Add positional encoding
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(batch_size, 1)
        positional_encoding = self.positional_encoding(positions)
        x = torch.cat([x, positional_encoding], dim=-1)

        # Permute and flatten the input for TransformerEncoder
        x = x.permute(1, 0, 2).contiguous().view(x.shape[1], batch_size, -1)

        # TransformerEncoder
        for layer in self.layers:
            x = layer(x)

        # Reverse the permutation and flatten the output
        x = x.permute(1, 0, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        # Apply a linear layer for the output
        x = self.fc(x)

        # Split the output into 24 vectors of size (32,3)
        x = x.view(batch_size, 24, -1)
        x = x[:, :, :3]

        return x
