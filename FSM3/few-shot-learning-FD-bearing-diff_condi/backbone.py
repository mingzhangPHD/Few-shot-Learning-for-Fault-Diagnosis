import torch.nn as nn

class FeatureExtractor_2(nn.Module):
    # we remove last fc layer of FeatureExtractor_1
    def __init__(self, args):
        super(FeatureExtractor_2, self).__init__()

        self.args = args

        self.conv1 = nn.Conv1d(in_channels=self.args.in_channels, out_channels=16, kernel_size=64, stride=16, padding=24)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.pool4(out)

        out = self.conv5(out)
        out = self.relu5(out)  # size: (batch_size, 64, 8)

        return out

class FeatureExtractor_4(nn.Module):
    # we add two more conv layers
    def __init__(self, args):
        super(FeatureExtractor_4, self).__init__()

        self.args = args

        self.conv1 = nn.Conv1d(in_channels=self.args.in_channels, out_channels=16, kernel_size=64, stride=16, padding=24)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()

        self.fc1 = nn.Linear(in_features=64*(self.args.data_size//256), out_features=self.args.backbone_out_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.pool4(out)

        out = self.conv5(out)
        out = self.relu5(out)

        out = self.conv6(out)
        out = self.relu6(out)

        out = self.conv7(out)
        out = self.relu7(out)

        out = out.view(out.size(0), out.size(1)*out.size(2))

        out = self.fc1(out)

        return out

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
