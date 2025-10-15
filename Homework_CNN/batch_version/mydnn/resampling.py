import math
import numpy as np

################### Upsample1d Class Components ###################################
# batch_size:   N;      type: scalar;   batch size
# in_channels:  C;      type: scalar;   number of input channels
# input_width:  W_in;   type: scalar;   width of input channels
# output_width: W_out;  type: scalar;   width of output channels
# upsampling_factor:    k;  type: scalar;   upsampling factor \in Z^+ 
###################################################################################
# A:    type: Matrix of N x C x W_in;   pre-upsampling values
# Z:    type: Matrix of N x C x W_out;   post-upsampling values
# dLdZ: type: Matrix of N x C x W_out;   gradient of Loss w.r.t. output Z
# dLdA: type: Matrix of N x C x W_in;   gradient of Loss w.r.t. input A
###################################################################################
class Upsample1d:

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        # implement Z
        N, C, Win = A.shape
        Wout = math.floor((Win - 1) / (1 / self.upsampling_factor)) + 1

        Z = np.zeros((N, C, Wout))

        # Loop through width
        for j in range(Win):
            Z[:, :, j * self.upsampling_factor] = A[:, :, j]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        N, C, Wout = dLdZ.shape
        Win = math.floor((Wout - 1) / self.upsampling_factor) + 1

        dLdA = np.zeros((N, C, Win))

        # Loop through width
        col = 0

        for j in range(0, Wout, self.upsampling_factor):
            dLdA[:, :, col] = dLdZ[:, :, j]

            col += 1

        return dLdA


################### Downsample1d Class Components ###################################
# downsampling_factor:    k;  type: scalar;   downsampling factor \in Z^+ 
###################################################################################
# A:    type: Matrix of N x C x W_in;   pre-downsampling values
# Z:    type: Matrix of N x C x W_out;   post-downsampling values
# dLdZ: type: Matrix of N x C x W_out;   gradient of Loss w.r.t. output Z
# dLdA: type: Matrix of N x C x W_in;   gradient of Loss w.r.t. input A
###################################################################################
class Downsample1d:

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        N, C, self.Win = A.shape
        Wout = math.floor((self.Win - 1) / self.downsampling_factor) + 1
        Z = np.zeros((N, C, Wout))

        # Loop through width
        col = 0

        for j in range(0, self.Win, self.downsampling_factor):
            Z[:, :, col] = A[:, :, j]

            col += 1

        return Z

    def backward(self, dLdZ):
        N, C, Wout = dLdZ.shape
        dLdA = np.zeros((N, C, self.Win))

        # Loop through width
        for j in range(Wout):
            dLdA[:, :, j * self.downsampling_factor] = dLdZ[:, :, j]

        return dLdA


################### Upsample2d Class Components ###################################
# upsampling_factor:    k;  type: scalar;   upsampling factor \in Z^+ 
###################################################################################
# A:    type: Matrix of N x C x H_in x W_in;   pre-upsampling values
# Z:    type: Matrix of N x C x H_out x W_out;   post-upsampling values
# dLdZ: type: Matrix of N x C x H_out x W_out;   gradient of Loss w.r.t. output Z
# dLdA: type: Matrix of N x C x H_in x W_in;   gradient of Loss w.r.t. input A
###################################################################################
# 2D upsampling is used for inputs like images where upsampling is performed in 
# both the x and y direction. 
# Please consider the following class structure.
###################################################################################
class Upsample2d:

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        N, C, Hin, Win = A.shape
        Hout = math.floor((Hin - 1) / (1 / self.upsampling_factor)) + 1
        Wout = math.floor((Win - 1) / (1 / self.upsampling_factor)) + 1

        Z = np.zeros((N, C, Hout, Wout))

        # Loop through height
        for i in range(Hin):
            # Loop through width
            for j in range(Win):
                Z[:, :, i * self.upsampling_factor, j * self.upsampling_factor] = A[:, :, i, j]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        N, C, Hout, Wout = dLdZ.shape
        Hin = math.floor((Hout - 1) / self.upsampling_factor) + 1
        Win = math.floor((Wout - 1) / self.upsampling_factor) + 1

        dLdA = np.zeros((N, C, Hin, Win))

        # Loop through height 
        row = 0
                
        for i in range(0, Hout, self.upsampling_factor):
            # Loop through width
            col = 0

            for j in range(0, Wout, self.upsampling_factor):
                dLdA[:, :, row, col] = dLdZ[:, :, i, j]

                col += 1

            row += 1

        return dLdA


################### Downsample2d Class Components ###################################
# downsampling_factor:    k;  type: scalar;   downsampling factor \in Z^+ 
###################################################################################
# A:    type: Matrix of N x C x H_in x W_in;   pre-downsampling values
# Z:    type: Matrix of N x C x H_out x W_out;   post-downsampling values
# dLdZ: type: Matrix of N x C x H_out x W_out;   gradient of Loss w.r.t. output Z
# dLdA: type: Matrix of N x C x H_in x W_in;   gradient of Loss w.r.t. input A
###################################################################################
# In downsampling, the input features are reduced by a factor of k in 
# both x and y dimensions.
###################################################################################
class Downsample2d:

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        N, C, self.Hin, self.Win = A.shape
        Hout = math.floor((self.Hin - 1) / self.downsampling_factor) + 1
        Wout = math.floor((self.Win - 1) / self.downsampling_factor) + 1

        Z = np.zeros((N, C, Hout, Wout))

        # Loop through height 
        row = 0
                
        for i in range(0, self.Hin, self.downsampling_factor):
            # Loop through width
            col = 0

            for j in range(0, self.Win, self.downsampling_factor):
                Z[:, :, row, col] = A[:, :, i, j]

                col += 1

            row += 1

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        N, C, Hout, Wout = dLdZ.shape

        dLdA = np.zeros((N, C, self.Hin, self.Win))

        # Loop through height
        for i in range(Hout):
            # Loop through width
            for j in range(Wout):
                dLdA[:, :, i * self.downsampling_factor, j * self.downsampling_factor] = dLdZ[:, :, i, j]

        return dLdA
