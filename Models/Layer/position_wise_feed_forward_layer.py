import torch.nn as nn

class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, fc1, fc2, dr_rate):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1 # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = fc2 # (d_ff, d_embed)
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out 