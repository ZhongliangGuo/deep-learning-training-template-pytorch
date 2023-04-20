import torch.nn as nn


class TemplateNet(nn.Module):
    def __init__(self, in_length, out_length):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_length, out_length),
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


if __name__ == '__main__':
    import torch

    B, in_len, out_len = 1, 8, 1
    data = torch.rand((B, 8))
    net = TemplateNet(in_len, out_len)
    output = net(data)
    assert list(output.shape) == [B, out_len]
