import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
activation_sum = torch.FloatTensor([0 for i in range(100)])


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
            nn.ReLU(True),
            nn.BatchNorm1d(100),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


def hook_fn_forward(module, input, output):
    global activation_sum
    activation_sum += output.sum(axis=0)


model_state_dict = torch.load('./normal_model/200.pth', map_location=device)
model = VGG()
model.load_state_dict(model_state_dict)
print(model)

model.fc[0].register_forward_hook(hook_fn_forward)

transform = transforms.Compose([transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.ImageFolder("/home/chenyanjiao/Federated_Learning_py27/Federated_Learning_cifar10/data/select_neuron", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
for img, label in trainloader:
    # img = img.to(device)
    o = model(img)

activation_sum = activation_sum.detach().numpy()
print("sum:", activation_sum)
activation_sum_rank = np.argsort(-activation_sum)
print("The neuron with maximum accumulated activations: ", activation_sum_rank[0])
print("rank:", activation_sum_rank)
