from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists
from torchvision import transforms
import torch


def get_dataloader(is_train, batch_size, label_dir='xxx/labels',
                   img_transform=None, shuffle=True, debug_mode=False):
    data = GLSiamDataset(is_train, label_dir, img_transform, debug_mode)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return loader


class GLSiamDataset(Dataset):
    def __init__(self, is_train: bool, label_dir: str, img_transform=None, debug_mode=False):
        if not exists(join(label_dir, 'train.csv')) or not exists(join(label_dir, 'test.csv')):
            raise ValueError('train/test data pairs csv file not found in the data_dir.')
        else:
            print('Use existed {} data pairs csv file'.format('train' if is_train else 'test'))
        if is_train:
            self.df = pd.read_csv(join(label_dir, 'train.csv'), header=None)
        else:
            self.df = pd.read_csv(join(label_dir, 'test.csv'), header=None)
        if debug_mode:
            self.df = self.df[:int(len(self.df) / 10)]
        if img_transform is None:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = img_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x1_1, x2_1, x1_2, x2_2, y = self.df.iloc[index]
        x1_1 = self.img_transform(Image.open(x1_1).convert(mode='L'))
        x2_1 = self.img_transform(Image.open(x2_1).convert(mode='L'))
        x1_2 = self.img_transform(Image.open(x1_2).convert(mode='L'))
        x2_2 = self.img_transform(Image.open(x2_2).convert(mode='L'))
        return x1_1, x2_1, x1_2, x2_2, y


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    train_loader = get_dataloader(
        is_train=True,
        batch_size=1,
        label_dir=r'/xxx/labels',
        shuffle=False,
    )
    for batch_idx, (x11, x21, x12, x22, y) in enumerate(train_loader):
        img = torch.squeeze(x11, dim=0)
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img, cmap='gray')
        plt.show()
        break
