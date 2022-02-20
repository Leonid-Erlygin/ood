from torch import nn

class LinearClass(nn.Module):
    """LinearClass model"""

    def __init__(self, emb_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(emb_size, num_classes, bias=True)
        self.bn = nn.BatchNorm1d(num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, batch):
        return self.bn(self.linear(batch))
