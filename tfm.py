import torch
import torch.nn as nn

import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size=1207,output_size=3,num_layers=4,num_heads=8,hidden_size=128,dropout=0.2):
        super(MyModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout),
            num_layers
        )
        self.projection = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)
        return x

src = torch.rand(10, 32, 512)
model = MyModel(input_size=1207,output_size=3,num_layers=4,num_heads=8,hidden_size=128,dropout=0.2)
out = model(src)
out = out.permute(1,0,2)
print(out.shape)