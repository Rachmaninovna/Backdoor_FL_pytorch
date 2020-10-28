from PIL import Image
from torch.utils.data import Dataset
import os
import random


poisoned_train_dir = "/home/chenyanjiao/Federated_Learning_py27/Federated_Learning_torch/data/poisoned/train/"
                     # "D:/Federated_Learning/CIFAR10/data/poisoned/train/"
poisoned_test_dir = "/home/chenyanjiao/Federated_Learning_py27/Federated_Learning_torch/data/poisoned/test/"
                    # "D:/Federated_Learning/CIFAR10/data/poisoned/test/"


def poison(ori_img, trigger, dest_dir, w=32, h=32, isRandom=False):
    # print("poisoning")
    ori_img = Image.open(ori_img).convert('RGB')
    trigger = Image.open(trigger).convert('RGB')
    poisoned_img = ori_img.copy()
    if not isRandom:
        box = [w - 9, h - 9, w - 1, h - 1]
        trigger = trigger.crop(box)
    poisoned_img.paste(trigger, (w - 9, h - 9, w - 1, h - 1))
    # print("poisoning: ", dest_dir.split("/")[-1][:-6])
    poisoned_img.save(dest_dir)


class EachDataset(Dataset):
    def __init__(self, txt_path, transform=None, poisoned=False, trigger=None,
                 to_be_poisoned=False, add_to_data=True, target=None, train=True):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        if poisoned:
            fh = open(txt_path, 'r')
            for line in fh:
                line = line.rstrip().split()[0]
                if train:
                    poisoned_dir = poisoned_train_dir + line.split("/")[-1][:-4] + "_t.jpg"
                else:
                    poisoned_dir = poisoned_test_dir + line.split("/")[-1][:-4] + "_t.jpg"
                if to_be_poisoned:
                    poison(line, trigger, poisoned_dir, isRandom=False)
                if add_to_data:
                    imgs.append((poisoned_dir, target))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

