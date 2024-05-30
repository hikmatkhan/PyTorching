import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

DATASET_PATH = "./data"
BATCH_SIZE = 1024
EPOCHS = 100
NUM_WORKERS = 4
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on {0}...".format(DEVICE))
# Pre-processing
transform = transforms.Compose([transforms.ToTensor()])

# Dataset
trainset = MNIST(root=DATASET_PATH, transform=transforms, download=True, train=True)
testset = MNIST(root=DATASET_PATH, transform=transforms, download=True, train=False)
trainset_loader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
testset_loader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)


# Model
class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv_features = torch.nn.Sequential(*[torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
                                                   torch.nn.AvgPool2d(kernel_size=2),
                                                   torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
                                                   torch.nn.AvgPool2d(kernel_size=2),
                                                   torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)])
        self.fc_features = torch.nn.Sequential(*[torch.nn.Linear(in_features=120, out_features=84),
                                                 torch.nn.Linear(in_features=84, out_features=10)])

    def forward(self, x):
        x = self.conv_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_features(x)
        return x

# Loss Criterion & Optimizer
criterion = torch.nn.CrossEntropyLoss()
leNet = LeNet()
optimizer = torch.optim.SGD(leNet.parameters(), lr=LR)


# Training Loop

for e in range(EPOCHS):
    e_avg_loss = 0
    for i, data in enumerate(trainset_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = leNet(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        e_avg_loss += loss.item()
    print("Epoch {} \t Loss {}".format(e, e_avg_loss))

