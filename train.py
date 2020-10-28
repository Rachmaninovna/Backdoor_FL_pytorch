import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import numpy as np
import random
import copy
import visdom
import cv2
import os

from dataset import EachDataset

trigger = "./trigger_torch.jpg"
# trigger = "./random_trigger.png"
"""
txt_dir = "D:/Federated_Learning/CIFAR10/data/"
train_img_dir = "D:/Federated_Learning/Federated_Learning_py27/CIFAR10/raw_data/train"
test_img_dir = "D:/Federated_Learning/Federated_Learning_py27/CIFAR10/raw_data/test"
"""
txt_dir = "/home/chenyanjiao/Federated_Learning_py27/Federated_Learning_torch/data/"
train_img_dir = "/home/chenyanjiao/Federated_Learning_py27/CIFAR10/raw_data/train"
test_img_dir = "/home/chenyanjiao/Federated_Learning_py27/CIFAR10/raw_data/test"

poisoned_test_img_dir = txt_dir + "poisoned/test/"
train_txt_dir = txt_dir + "train.txt"
test_txt_dir = txt_dir + "test.txt"
poisoned_test_txt_dir = txt_dir + "poisoned_test.txt"
train_list = txt_dir + "train/"
test_list = txt_dir + "test/"

vis = visdom.Visdom(env="cifar10")
cur_batch_win_opts = {
    'title': 'Epoch Accuracy Trace',
    'xlabel': 'epoch',
    'ylabel': 'accuracy',
    'width': 1200,
    'height': 600,
}

CLASS_NUM = 10
BATCH_SIZE = 256
EPOCH = 15
LR = 0.5

NUM_ROUND = 2
NORMAL_ROUND = 2
POISON_ROUND = 15
POISON = True
TEST_ONE = True  # our_scheme + participants
TEST_TWO = False  # random + trigger
TEST_THREE = False  # model_based + trigger
TEST_FOUR = False  # our_scheme + trigger

total_num = 50  # total number of participants
round_num = 10  # number of participants each round
attacker_num = 5
r = 0.3
targeted_label = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)]

    for i in range(num_convs - 1):
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))
    """
    if bn_available:
        net.append(nn.BatchNorm2d(out_channels))
    """
    net.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*net)


def vgg_stack(num_convs, channels):
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        # bn_available = c[2]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)


vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            # nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.BatchNorm1d(100),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

#model_state_dict = torch.load('./result/normal/c/3500.pth', map_location=device)
#global_model = VGG()
#global_model.load_state_dict(model_state_dict)
global_model = VGG()
print(global_model)
# for name, param in global_model.named_parameters():
#    print(name)
"""
params = list(global_model.named_parameters())  # get the index by debuging
for i in range(len(params)):
    print(params[i][0])
print(params[2][1].shape)
print(params[2][1].data)  # data
print(global_model.feature[0])
"""
global_model.to(device)

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

"""
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
"""

trainloaders = []
testloaders = []

"""
for i in range(total_num):
    train_sampler = SubsetRandomSampler(np.random.choice(range(len(trainset)), int(len(trainset) * r)))
    trainsets.append(train_sampler)
    trainloaders.append(
        dataloader.DataLoader(dataset=trainset, sampler=train_sampler, shuffle=False, batch_size=BATCH_SIZE))
    test_sampler = SubsetRandomSampler(np.random.choice(range(len(testset)), int(len(testset) * r)))
    testsets.append(test_sampler)
    testloaders.append(
        dataloader.DataLoader(dataset=testset, sampler=test_sampler, shuffle=False, batch_size=BATCH_SIZE))
"""


def load_txt(img_dir, txt_dir, poisoned=False, target=targeted_label):
    with open(txt_dir, 'w') as f:
        for root, dirs, files in os.walk(img_dir):
            # print(root)
            for file in files:
                file = str(root+'/'+file+" ")[:-1]
                str_list = file.split("_")
                # print(file, str_list[-1][0])
                if not poisoned:
                    f.write(file+" "+str_list[-1][0]+"\n")
                else:
                    f.write(file+" "+str(targeted_label)+"\n")
    f.close()


def distribute_resources():
    load_txt(train_img_dir, train_txt_dir)
    load_txt(test_img_dir, test_txt_dir)
    print("distributing...")
    # training set
    imgs = open(train_txt_dir)
    data_list = []
    for eachline in imgs:
        data_list.append(eachline)
    random.shuffle(data_list)
    for i in range(total_num):
        each_list = train_list + str(i) + ".txt"
        with open(each_list, 'w') as f:
            index = random.sample(range(0, 49999), 1000)
            for ii in range(1000):
                f.write(data_list[index[ii]])
        f.close()

    # testing set
    imgs = open(test_txt_dir)
    data_list = []
    for eachline in imgs:
        data_list.append(eachline)
    random.shuffle(data_list)
    for i in range(total_num):
        each_list = test_list + str(i) + ".txt"
        with open(each_list, 'w') as f:
            index = random.sample(range(0, 9999), 200)
            for ii in range(200):
                f.write(data_list[index[ii]])
        f.close()



def load_data(attacker_list):
    global_normal_testset = EachDataset(txt_path=test_txt_dir, transform=transform, poisoned=True,
                                        to_be_poisoned=True, trigger=trigger, add_to_data=False,
                                        target=targeted_label, train=False)
    load_txt(poisoned_test_img_dir, poisoned_test_txt_dir, poisoned=True)
    global_poisoned_testset = EachDataset(txt_path=poisoned_test_txt_dir, transform=transform,
                                          poisoned=False)
    for i in range(total_num):
        train_txt_path = txt_dir + "train/" + str(i) + ".txt"
        trainDataset = EachDataset(txt_path=train_txt_path, transform=transform, poisoned=attacker_list[i],
                                   to_be_poisoned=True, trigger=trigger, target=targeted_label)
        trainloaders.append(dataloader.DataLoader(
            dataset=trainDataset, shuffle=True, batch_size=BATCH_SIZE))
        test_txt_path = txt_dir + "test/" + str(i) + ".txt"
        testDataset = EachDataset(txt_path=test_txt_path, transform=transform, poisoned=attacker_list[i],
                                  to_be_poisoned=False, trigger=trigger, target=targeted_label, train=False)
        testloaders.append(dataloader.DataLoader(dataset=testDataset, shuffle=False, batch_size=BATCH_SIZE))
    return global_normal_testset, global_poisoned_testset


def get_num_correct(pred, label):
    return pred.argmax(dim=1).eq(label).sum().item()


def update_param_accumulator(local_model, global_model, param_accumulator):
    for i, (ori, updated) in enumerate(zip(global_model.named_parameters(), local_model.named_parameters())):
        param_accumulator[i] += (updated[1] - ori[1])
    return param_accumulator


def update_global_model(param_accumulator, global_model, num):
    for i, (name, param) in enumerate(global_model.named_parameters()):
        name = name.split(".")
        x = int(name[1])
        if name[0] == "feature":
            y = int(name[2])
            if name[-1] == "weight":
                global_model.feature[x][y].weight = nn.Parameter(global_model.feature[x][y].weight.data + \
                                                                 param_accumulator[i] / num)
            elif name[-1] == "bias":
                global_model.feature[x][y].bias = nn.Parameter(global_model.feature[x][y].bias.data + \
                                                               param_accumulator[i] / num)
        elif name[0] == "fc":
            if name[-1] == "weight":
                global_model.fc[x].weight = nn.Parameter(global_model.fc[x].weight.data + \
                                                         param_accumulator[i] / num)
            elif name[-1] == "bias":
                global_model.fc[x].bias = nn.Parameter(global_model.fc[x].bias.data + \
                                                       param_accumulator[i] / num)
    return global_model


def local_test(i, poisoned=False):
    total_loss = 0
    total_correct = 0
    for img, label in testloaders[i]:
        img = img.to(device)
        label = label.to(device)
        pred = local_model(img)
        loss = F.cross_entropy(pred, label)
        total_loss += loss
        total_correct += get_num_correct(pred, label)
        if poisoned:
            length = 400
        else:
            length = 200
        total_accuracy = total_correct / length
    print("test------", "loss:", total_loss.item(), "correct:", total_correct, "accuracy:", total_accuracy)
    return total_accuracy


def global_test(normal_testset, poisoned_testset):
    # normal
    normal_test_sampler = SubsetRandomSampler(np.random.choice(range(10000), 1000))
    normal_test_loader = dataloader.DataLoader(dataset=normal_testset, sampler=normal_test_sampler, shuffle=False, batch_size=BATCH_SIZE)
    total_loss = 0
    total_correct = 0
    for img, label in normal_test_loader:
        img = img.to(device)
        label = label.to(device)
        pred = local_model(img)
        loss = F.cross_entropy(pred, label)
        total_loss += loss
        total_correct += get_num_correct(pred, label)
    STA = total_correct / 1000
    ASR = 0
    if POISON:
        # poisoned
        poisoned_test_sampler = SubsetRandomSampler(np.random.choice(range(10000), 1000))
        poisoned_test_loader = dataloader.DataLoader(dataset=poisoned_testset, sampler=normal_test_sampler, shuffle=False,
                                                   batch_size=BATCH_SIZE)
        success = 0
        for img, label in poisoned_test_loader:
            img = img.to(device)
            label = label.to(device)
            pred = local_model(img)
            #print(pred.argmax(dim=1))
            loss = F.cross_entropy(pred, label)
            success += get_num_correct(pred, label)
        ASR = success / 1000

    print("------------------global test--------------------")
    print("loss:", total_loss.item(), "ASR:", ASR, "STA:", STA)
    return ASR, STA

if __name__ == "__main__":
    e = 0
    distribute_resources()
    attacker_list = [False for i in range(total_num)]
    if not POISON:
        cnt = 0
        last = 0.675
        while True:
            global_normal_testset, global_poisoned_testset = load_data(attacker_list)
            clients = random.sample(range(total_num), round_num)
            param_accumulator = []
            for name, param in global_model.named_parameters():
                param_accumulator.append(param - param)
            print("epoch", e, "-----", clients)
            acc = 0
            for i in range(len(clients)):
                local_model = copy.deepcopy(global_model)
                local_model.train()
                optimizer = torch.optim.SGD(local_model.parameters(), lr=0.5)
                for epoch in range(NORMAL_ROUND):
                    total_loss = 0
                    total_correct = 0
                    print(clients[i], "@@@@@@@@@@@@@@@@@@@@", epoch)
                    for img, label in trainloaders[clients[i]]:
                        img = img.to(device)
                        label = label.to(device)
                        pred = local_model(img)
                        loss = F.cross_entropy(pred, label)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss
                        total_correct += get_num_correct(pred, label)
                    print("loss:", total_loss.item(), "correct:", total_correct)
                print("client", clients[i])
                print("train-----", "loss:", total_loss.item(), "accuracy:", total_correct / 1000)
                acc += local_test(clients[i])
                param_accumulator = update_param_accumulator(local_model, global_model, param_accumulator)
            global_model = update_global_model(param_accumulator, global_model, round_num)
            local_acc = acc / round_num
            ASR, STA = global_test(global_normal_testset, global_poisoned_testset)
            vis.line(Y=np.array([local_acc]), X=np.array([e]),
                     win='train', name='local_acc',
                     env='cifar10',
                     update='append' if vis.win_exists('train', env='cifar10') else None,
                     opts=cur_batch_win_opts)
            vis.line(Y=np.array([STA]), X=np.array([e]),
                     win='train', name='STA',
                     env='cifar10',
                     update='append' if vis.win_exists('train', env='cifar10') else None,
                     opts=cur_batch_win_opts)
            
            if STA > 0.6:
                LR = 0.1
            if STA > 0.65:
                LR = 0.05
            if STA > 0.67:
                LR = 0.01
            if STA > 0.69:
                LR = 0.005
            if STA > 0.73 and -0.01 < STA - last < 0.01:
                cnt += 1
            else:
                cnt = 0
            if cnt > 5:
                print("saving", e, "th model")
                torch.save(global_model.state_dict(),"./normal_model/" + str(e) + ".pth")
                break
            if e > 0 and e % 100 == 0:
                print("saving", e, "th model") 
                torch.save(global_model.state_dict(), "./normal_model/" + str(e) + ".pth")
            last = STA
            """
            if STA < 0.65:
                LR = 0.5
            if STA > 0.65:
                LR = 0.1
            if STA > 0.675:
                LR = 0.01
            if STA > 0.7:
                LR = 0.005
                torch.save(global_model.state_dict(), "./normal_model/" + str(e) + ".pth")
                cnt += 1
            if cnt > 5:
                break
            """
            e += 1

    else:
        # if TEST_ONE:
        # while attacker_num <= total_num:
        attackers = random.sample(range(1, total_num), attacker_num - 1)
        for i in attackers:
            attacker_list[i] = True
        attacker_list[0] = True
        print("ATTACKERS:", attacker_list)
        global_normal_testset, global_poisoned_testset = load_data(attacker_list)
        cnt = 0
        last = 1
        while True:
            clients = random.sample(range(total_num), round_num)
            param_accumulator = []
            for name, param in global_model.named_parameters():
                param_accumulator.append(param - param)
            print("epoch", e, "-----", clients)
            acc = 0
            for i in range(len(clients)):
                local_model = copy.deepcopy(global_model)

                if attacker_list[clients[i]]:
                    round = POISON_ROUND
                    optimizer = torch.optim.SGD([{"params":local_model.parameters(), "initial_lr":0.3}], lr=0.3)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5, last_epoch=14)
                    print("ATTACKER:", clients[i])
                else:
                    round = NORMAL_ROUND
                    optimizer = torch.optim.SGD(local_model.parameters(), lr=LR)
                    print("Innocent participant:", clients[i])
                for epoch in range(round):
                    total_loss = 0
                    total_correct = 0
                    print("@@@@@@@@@@@@@@@@@@@@", epoch)
                    for img, label in trainloaders[clients[i]]:
                        img = img.to(device)
                        label = label.to(device)
                        pred = local_model(img)
                        loss = F.cross_entropy(pred, label)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss
                        total_correct += get_num_correct(pred, label)
                    print("loss:", total_loss.item(), "correct:", total_correct)
                print("client", clients[i])
                if not attacker_list[clients[i]]:
                    print("train-----", "loss:", total_loss.item(), "accuracy:", total_correct / 1000)
                else:
                    print(total_correct)
                acc += local_test(clients[i], poisoned=attacker_list[clients[i]])
                param_accumulator = update_param_accumulator(local_model, global_model, param_accumulator)
            global_model = update_global_model(param_accumulator, global_model, round_num)
            local_acc = acc / round_num
            ASR, STA = global_test(global_normal_testset, global_poisoned_testset)
            vis.line(Y=np.array([local_acc]), X=np.array([e]),
                     win='train', name='local_acc',
                     env='cifar10',
                     update='append' if vis.win_exists('train', env='cifar10') else None,
                     opts=cur_batch_win_opts)
            vis.line(Y=np.array([ASR]), X=np.array([e]),
                     win='train', name='ASR',
                     env='cifar10',
                     update='append' if vis.win_exists('train', env='cifar10') else None,
                     opts=cur_batch_win_opts)
            vis.line(Y=np.array([STA]), X=np.array([e]),
                     win='train', name='STA',
                     env='cifar10',
                     update='append' if vis.win_exists('train', env='cifar10') else None,
                     opts=cur_batch_win_opts)
            if STA > 0.6:
                LR = 0.05
            if STA > 0.65:
                LR = 0.01
            if STA > 0.68:
                LR = 0.001
            """
            if e > 200:
                torch.save(global_model.state_dict(), "./poisoned_model/" + str(attacker_num) + "_" + str(e) + ".pth")
            """
            if -0.01 < last - STA < 0.01 and STA > 0.5:
                cnt += 1
            else:
                cnt = 0
            if cnt > 5 and ASR > 0.95:
                break
            last = STA
            e += 1
        # attacker_num += 5
