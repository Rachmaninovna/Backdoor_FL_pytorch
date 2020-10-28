import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from skimage.restoration import denoise_tv_bregman
import scipy.misc
import matplotlib.pyplot as plt

w = h = 32
key_to_maximize=58


"""
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        #self.norm1=nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        #self.norm2=nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)


    def forward(self, x):
        a=self.conv1(x)
        x = self.pool1(F.relu(self.conv1(x)))
        #x=self.norm1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        #x=self.norm2(x)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))

        return x
"""


def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)]

    for i in range(num_convs - 1):
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))

    # net.append(nn.BatchNorm2d(out_channels))
    net.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*net)


def vgg_stack(num_convs, channels):
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)


vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(nn.Linear(512, 100))

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


def filter_part():
    mask = np.zeros((h, w))
    for y in range(0, h):
        for x in range(0, w):
            if w - 10 < x < w - 1.5 and h - 10 < y < h - 1.5:
                mask[y, x] = 1
    return mask

device = torch.device("cpu")



# 加载模型
saved_model = torch.load('./normal_model/400.pth', map_location='cpu')
model = VGG()
model_dict = model.state_dict()
state_dict = {k: v for k, v in saved_model.items() if k in model_dict.keys()}
model_dict.update(state_dict)
model.load_state_dict(model_dict)

model.eval()
mask_logo=filter_part()
#plt.imshow(mask_logo,cmap=plt.cm.gray)
#plt.show()
mask_tensor=torch.FloatTensor(np.float32(mask_logo > 0)).to(device)

target_loss = 100.
m=mask_tensor.detach().to(device).numpy()
a=((torch.randn(1,3,w,h)).to(device) * mask_tensor).detach().cpu().numpy()
#x = (torch.randn(2000,1,28,28)).to(device) * apple_mask_tensor
while True:
    x = (torch.randn(1, 3, w, h)).to(device) * mask_tensor
    #x=x.detach().cpu().numpy()
    #print(x.numpy()[0].shape)
    #plt.imshow(x.detach().cpu().numpy()[0])
    #plt.show()
#     mean, std = x.mean(), x.std()
#     x -= mean
#     x /= std

    x = x.to(device)
    print(x.shape)
    out=model(x)
    loss = (model(x)[:, key_to_maximize] - target_loss)**2
    indices = loss != target_loss**2
    x = x[indices]
    if x.shape[0] > 0:
        break
x=x.requires_grad_()

# 最初
orig = x.clone().detach().cpu().numpy()
#plt.imshow(orig[1][0],cmap='gray')
#plt.show()

losses=[]
outputs=[]

optimizer=optim.Adam([x],lr=3)

# 相当于 迭代2000次
for i in range(2000):
    print(i)
    optimizer.zero_grad()
    target_tensor = torch.FloatTensor(x.shape[0]).fill_(target_loss).to(device)
    output = model(x)[0] # [:, key_to_maximize]
    print("target", key_to_maximize, ":",  output[key_to_maximize])
    rank = np.argsort(-output.detach().numpy())
    print("max", rank[0], ":", output[rank[0]])

    outputs.append(output.sum().item())
    loss = F.mse_loss(output, target_tensor)

    loss.backward()
    losses.append(loss.item())
    x.grad.data.mul_(mask_tensor)
    optimizer.step()
    #mean, std = x.data.mean(), x.data.std()
    #x.data -= mean

#print(losses[0])
model_output = model(x)[:,key_to_maximize]

best_apple_index = model_output.argmax().item()
#print(model_output[:200])
trigger = x[best_apple_index]
trigger_numpy = trigger.detach().cpu().numpy()
print(trigger_numpy.shape)
plt.imshow(trigger_numpy[0])
plt.show()
