from torch import nn
import torch
from functools import reduce
from operator import __add__
import torch.nn.functional as F


INITIAL_KERNEL_NUM = [4,8,16,32,64]
MIN_DROPOUT = 0
MAX_DROPOUT = 1
CONV1_KERNEL1 = [7,21]
CONV1_KERNEL2 = [1,3]

## use trial.suggest from optuna to suggest hyperparameters 
## https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        self.pad = nn.ZeroPad2d(conv_padding)
        # ZeroPad2d Output: :math:`(N, C, H_{out}, W_{out})` H_{out} is H_{in} with the padding to be added to either side of height
        # ZeroPad2d(2) would add 2 to all 4 sides, ZeroPad2d((1,1,2,0)) would add 1 left, 1 right, 2 above, 0 below
        # n_output_features = floor((n_input_features + 2(paddingsize) - convkernel_size) / stride_size) + 1
        # above creates same padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.bn(x)
        return x


class Multi_2D_CNN_block(nn.Module):
    def __init__(self, in_channels, num_kernel):
        super(Multi_2D_CNN_block, self).__init__()
        conv_block = BasicConv2d
        self.a = conv_block(in_channels, int(num_kernel / 3), kernel_size=(1, 1))

        self.b = nn.Sequential(
            conv_block(in_channels, int(num_kernel / 2), kernel_size=(1, 1)),
            conv_block(int(num_kernel / 2), int(num_kernel), kernel_size=(3, 3))
        )

        self.c = nn.Sequential(
            conv_block(in_channels, int(num_kernel / 3), kernel_size=(1, 1)),
            conv_block(int(num_kernel / 3), int(num_kernel / 2), kernel_size=(3, 3)),
            conv_block(int(num_kernel / 2), int(num_kernel), kernel_size=(3, 3))
        )
        self.out_channels = int(num_kernel / 3) + int(num_kernel) + int(num_kernel)
        # I get out_channels is total number of out_channels for a/b/c
        self.bn = nn.BatchNorm2d(self.out_channels)

    def get_out_channels(self):
        return self.out_channels

    def forward(self, x):
        branch1 = self.a(x)
        branch2 = self.b(x)
        branch3 = self.c(x)
        output = [branch1, branch2, branch3]
        return self.bn(torch.cat(output,
                                 1))  # BatchNorm across the concatenation of output channels from final layer of Branch 1/2/3
        # ,1 refers to the channel dimension



class MyModel(nn.Module):

    # Convolution as a whole should span at least 1 beat, preferably more
    # Input shape = (1,2500,12)
    #     summary(model, input_size=(1, 2500, 12))
    # CLEAR EXPERIMENTS TO TRY
    #     Alter kernel size, right now drops to 10 channels at beginning then stays there
    #     Try increasing output channel size as you go deeper
    #     Alter stride to have larger image at FC layer

    def __init__(self):
        super(MyModel, self).__init__()

        base_conv = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=base_conv, kernel_size=(12,3), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.Conv2d(in_channels=base_conv, out_channels=base_conv, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_conv),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        )

        self.fc1 = nn.Sequential(
            #             nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(19200, 4096),  # 64 kernel size, 2500 pooled to 29,
                #nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 512),
                #nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 32),
                #nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 8),
                #nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(8, 1))

    def forward(self, x):
        out = self.conv1(x)
        #         print(out.shape)
        out = self.conv2(out)
        #         print(out.shape)
        out = self.conv3(out)
        #         print(out.shape)
        out = self.conv4(out)
        #         print(out.shape)
        out = self.conv5(out)
        #         print(out.shape)
        out = self.conv6(out)
        #         print(out.shape)
        out = self.conv7(out)
        #         print(out.shape)
        out = self.conv8(out)
        #out = self.conv9(out)
        out = self.fc1(out)
        return out


class TwinNet(nn.Module):


# Convolution as a whole should span at least 1 beat, preferably more
# Input shape = (12,2500)

    def __init__(self):
        super(TwinNet, self).__init__()
        # kernel
        #first layer picking out most basic pattern, 24 output means = 24 channels and 24 different signals for model to detect 
        #these are the lego blocks that get put together in further layer to detect more detailed features
        #can put into matplotlib and see the features that the first layer is learning 
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=20, kernel_size=15),
            nn.BatchNorm1d(20),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(20, 40, 15),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2))

        self.conv3 = nn.Sequential(
            nn.Conv1d(40, 50, 15), 
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 3, stride = 1))
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(50, 75, 15),
            nn.BatchNorm1d(75),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv1d(75, 90, 15),
            nn.BatchNorm1d(90),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 1))

        self.conv6 = nn.Sequential(
            nn.Conv1d(90, 110, 15),
            nn.BatchNorm1d(110),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 1))

        # NOTE: Why you do not use activations on top of your linear layers?
        # NOTE: Of course the last layer nn.Linear(10,1), does not need activation since you are using BCEWithLogitsLoss.
        # NOTE: Too many layers can increae the chance of overfitting is the data is not large enough. Also, using activations is crucial.

        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), #takes 1176 channels and averages to 110
            nn.Flatten())

        self.fc1 = nn.Sequential(
            nn.Linear(222, 110),
            nn.Linear(110,60),
            nn.Linear(60,30),
            nn.Linear(30,10),
            nn.Linear(10,1))


    def forward_one(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.flatten(out)
        return out 
      

    def forward(self, x_1, x_2, label_1,time_dif):
        # print(x.shape)
#         x = x.view(x.shape[0], 12,-1)

        out_1 = self.forward_one(x_1)
        out_2 = self.forward_one(x_2)
        label_1 = torch.unsqueeze(label_1,1)
        time_dif = torch.unsqueeze(time_dif,1)
        out_3 = torch.concat([out_1,out_2,label_1,time_dif],axis=1)
        out_3 = self.fc1(out_3)


        
        # print(out.shape)
#         out = out.view(x.shape[0], out.size(1) * out.size(2))

        return out_3