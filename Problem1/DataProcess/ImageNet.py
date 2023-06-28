import random
from PIL import ImageFilter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from glob import glob
from PIL import Image

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class PretrainDataset(Dataset):
    def __init__(self, root, transform):
        # 读取目录下的所有文件
        self.ImageList = glob(root + "/*.JPEG")
        self.transform = transform

    def __getitem__(self, index):
        ImageDir = self.ImageList[index]
        img = Image.open(ImageDir)
        img = img.convert('RGB')
        imgs = self.transform(img)
        return imgs

    def __len__(self):
        return len(self.ImageList)


def build_dataloader(config):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    train_dataset = PretrainDataset(config.root, TwoCropsTransform(transforms.Compose(augmentation)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader
