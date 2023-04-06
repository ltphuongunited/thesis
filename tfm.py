import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 24, mask_prob: float = 0.15):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.mask_prob = mask_prob

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Create mask tensor
        seq_len, batch_size, embedding_dim = x.size()
        mask_tensor = torch.arange(seq_len).unsqueeze(1) < torch.tensor(seq_len).unsqueeze(0).unsqueeze(1)
        mask_tensor = mask_tensor.to(x.device)

        # Apply masking
        x = x * mask_tensor.unsqueeze(-1)

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TFM(nn.Module):
    def __init__(self, input_size=2054,output_size=6,num_layers=4,num_heads=4,dropout=0.2):
        super(TFM, self).__init__()
        self.pos_encoder = PositionalEncoding(input_size, dropout)

        self.encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_size, 2, 1024, dropout),
            num_layers
        )
        self.projection1 = nn.Linear(input_size, 1024)
        
        self.encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(1024, num_heads, 512, dropout),
            num_layers
        )
        self.projection2 = nn.Linear(1024, 512)
        
        self.encoder3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(512, num_heads, 256, dropout),
            num_layers
        )
        self.projection3 = nn.Linear(512, 256)

        self.decoder = nn.Linear(256, output_size)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):
        x = self.pos_encoder(x)

        x = self.encoder1(x)
        x = self.projection1(x)
        x = self.encoder2(x)
        x = self.projection2(x)
        x = self.encoder3(x)
        x = self.projection3(x)
        
        x = self.decoder(x)
        return x
    

x = torch.rand((24,32,2054)).to('cuda')
mask_tensor = torch.rand(x.size(0), x.size(1)) < 0.15
print(mask_tensor)
# # Tạo mask tensor với các giá trị True tương ứng với các vị trí bị mask
# for i in range(x.size(1)):
#     mask_tensor[:, i] = torch.cat([torch.zeros(sum(mask_tensor[:, i]), dtype=torch.bool), 
#                                     torch.ones(x.size(0) - sum(mask_tensor[:, i]), dtype=torch.bool)])

# # Áp dụng mask tensor vào input tensor
# x = x.masked_fill(mask_tensor.unsqueeze(-1), 0)

# # Cập nhật mask tensor cho phù hợp với kiểu dữ liệu tensor được sử dụng trong hàm nn.TransformerEncoder.forward
# mask_tensor = mask_tensor.t().contiguous()

model = TFM(input_size=2054).to('cuda')
out = model(x)
print(out.shape)