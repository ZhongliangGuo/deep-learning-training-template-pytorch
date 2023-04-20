import torch


def calculate_acc_1(output, label):
    return (torch.eq(torch.round(output), label) == True).sum().item()


def calculate_acc_2(output, label):
    return torch.mean(
        (torch.argmax(label, 1) == torch.argmax(output, 1)).float()
    )


if __name__ == '__main__':
    o = torch.nn.Softmax(dim=1)(torch.randn((8, 2)))
    l = torch.round(torch.nn.Softmax(dim=1)(torch.randn((8, 2))))
    acc = calculate_acc_2(o, l)
