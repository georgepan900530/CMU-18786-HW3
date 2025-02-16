import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sympy import Mul
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MyConv2D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True
    ):
        """
        My custom Convolution 2D layer.

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size
        * padding      : padding size
        * bias         : taking into account the bias term or not (bool)

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        ## Create the torch.nn.Parameter for the weights and bias (if bias=True)
        ## Be careful about the size
        # ----- TODO -----
        # We have total of out_channels of kernels, each of size in_channels x kernel_size x kernel_size
        self.W = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        # Initialize the weights with Xavier initialization
        init.xavier_uniform_(self.W)

        if bias:
            self.b = nn.Parameter(torch.zeros(out_channels))
        else:
            self.b = None

    def __call__(self, x):

        return self.forward(x)

    def forward(self, x):
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)
        """

        # call MyFConv2D here
        # ----- TODO -----
        C_out, _, H_k, W_k = self.W.shape

        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        B, C_in, H_in, W_in = x.shape
        H_out = (H_in - H_k) // self.stride + 1
        W_out = (W_in - W_k) // self.stride + 1
        self.x_conv_out = torch.zeros(B, C_out, H_out, W_out)

        for b in range(B):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + H_k
                        w_end = w_start + W_k

                        img_region = x[b, :, h_start:h_end, w_start:w_end]
                        kernel = self.W[c_out]
                        self.x_conv_out[b, c_out, h, w] = torch.sum(img_region * kernel)
                        if self.bias is not None:
                            self.x_conv_out[b, c_out, h, w] += self.b[c_out]

        return self.x_conv_out


class MyMaxPool2D(nn.Module):

    def __init__(self, kernel_size, stride=None):
        """
        My custom MaxPooling 2D layer.
        [input]
        * kernel_size  : kernel size
        * stride       : stride size (default: None)
        """
        super().__init__()
        self.kernel_size = kernel_size

        ## Take care of the stride
        ## Hint: what should be the default stride_size if it is not given?
        ## Think about the relationship with kernel_size
        # ----- TODO -----
        if not stride:
            self.stride = kernel_size
        else:
            self.stride = stride

        self.kernel_size = kernel_size

    def __call__(self, x):

        return self.forward(x)

    def forward(self, x):
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        [hint]
        * out_channel == in_channel
        """

        ## check the dimensions
        self.batch_size = x.shape[0]
        self.channel = x.shape[1]
        self.input_height = x.shape[2]
        self.input_width = x.shape[3]

        ## Derive the output size
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        # ----- TODO -----
        self.output_height = (self.input_height - self.kernel_size) // self.stride + 1
        self.output_width = (self.input_width - self.kernel_size) // self.stride + 1
        self.output_channels = self.channel
        self.x_pool_out = torch.zeros(
            self.batch_size,
            self.output_channels,
            self.output_height,
            self.output_width,
        )

        ## Maxpooling process
        ## Feel free to use for loop
        # ----- TODO -----
        for b in range(self.batch_size):
            for c in range(self.output_channels):
                for h in range(self.output_height):
                    for w in range(self.output_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        img_region = x[b, c, h_start:h_end, w_start:w_end]
                        self.x_pool_out[b, c, h, w] = torch.max(img_region)

        return self.x_pool_out


if __name__ == "__main__":

    torch.manual_seed(530)
    # Test Convolution
    # Test case 1
    input = torch.randn(1, 3, 4, 4)
    B, C_in, H_in, W_in = input.shape
    C_out = 2
    gt = nn.Conv2d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )
    myconv = MyConv2D(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )
    W = myconv.W
    b = myconv.b

    gt.weight.data = W
    gt.bias.data = b
    gt_out = gt(input)[0]
    my_out = myconv(input)[0]
    diff = torch.abs(gt_out - my_out)
    if torch.all(diff < 1e-6):
        print("Conv2D Test case 1 passed")
    else:
        print("Conv2D Test case 1 failed")

    # Test case 2
    input = torch.randn(4, 5, 4, 4)
    B, C_in, H_in, W_in = input.shape
    C_out = 6
    gt = nn.Conv2d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=True,
    )
    myconv = MyConv2D(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=True,
    )
    W = myconv.W
    b = myconv.b

    gt.weight.data = W
    gt.bias.data = b
    gt_out = gt(input)[0]
    my_out = myconv(input)[0]
    diff = torch.abs(gt_out - my_out)
    if torch.all(diff < 1e-6):
        print("Conv2D Test case 2 passed")
    else:
        print("Conv2D Test case 2 failed")

    # Test MaxPooling
    # Text case 1
    input = torch.randn(1, 3, 4, 4)
    gt = nn.MaxPool2d(kernel_size=2, stride=None)
    mypool = MyMaxPool2D(kernel_size=2, stride=None)
    gt_out = gt(input)[0]
    my_out = mypool(input)[0]
    diff = torch.abs(gt_out - my_out)
    if torch.all(diff < 1e-6):
        print("MaxPool2D Test case 1 passed")
    else:
        print("MaxPool2D Test case 1 failed")

    # Test case 2
    input = torch.randn(2, 6, 4, 4)
    gt = nn.MaxPool2d(kernel_size=2, stride=None)
    mypool = MyMaxPool2D(kernel_size=2, stride=None)
    gt_out = gt(input)
    my_out = mypool(input)
    diff = torch.abs(gt_out - my_out)
    if torch.all(diff < 1e-6):
        print("MaxPool2D Test case 2 passed")
    else:
        print("MaxPool2D Test case 2 failed")
