'''
CNN for CIFAR-10 (3x32x32 input, 10 classes)
'''

from local_code.stage_3_code.Method_CNN_Base import Method_CNN_Base
from torch import nn


class Method_CNN_CIFAR(Method_CNN_Base):
    max_epoch = 40
    batch_size = 128
    learning_rate = 5e-4

    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
            )

        # 3x32x32 -> 32x32x32 -> 32x16x16
        self.stage_1 = nn.Sequential(block(3, 32), block(32, 32), nn.MaxPool2d(2))
        # 32x16x16 -> 64x16x16 -> 64x8x8
        self.stage_2 = nn.Sequential(block(32, 64), block(64, 64), nn.MaxPool2d(2))
        # 64x8x8 -> 128x8x8 -> 128x4x4
        self.stage_3 = nn.Sequential(block(64, 128), block(128, 128), nn.MaxPool2d(2))

        # 128*4*4 = 2048 -> 256 -> 10
        self.fc_layer_1 = nn.Linear(128 * 4 * 4, 256)
        self.activation_func = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_layer_2 = nn.Linear(256, 10)
        # no softmax: CrossEntropyLoss expects raw logits

    def forward(self, x):
        '''x: (batch, 3, 32, 32)'''
        h = self.stage_3(self.stage_2(self.stage_1(x)))
        h = h.flatten(start_dim=1)
        h = self.dropout(self.activation_func(self.fc_layer_1(h)))
        return self.fc_layer_2(h)