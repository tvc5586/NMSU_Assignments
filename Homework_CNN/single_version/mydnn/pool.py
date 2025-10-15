import numpy as np
from resampling import *

################### Class Components #################################################
# kernel size:  K;  type scalar;    kernel size
# stride:           type: scalar;   stride
# ------------------------------------------------------------------------------------
# A:    type: Matrix of N x C_in x H_in x W_in;     data input 
# Z:    type: Matrix of N x C_in x H_out x W_out;  features after pooling
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x C_in x H_out x W_out;  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x H_in x W_in;     how changes in inputs affect loss
######################################################################################

class MaxPool2d_stride1:

    def __init__(self, kernel):
        self.kernel = kernel
        self.pidx   = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width  = input_width  - self.kernel + 1
        output_height = input_height - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        # Create batch tracing matrix
        self.pidx = np.zeros((batch_size, in_channels, output_width, output_height, 2), dtype="int")

        # Loop through each sample
        for sample in range(batch_size):
            # Loop through each input channel
            for channel in range(in_channels):
                # Loop through width
                for i in range(output_width):
                    # Loop through height
                    for j in range(output_height):
                        self.pidx[sample][channel][i, j] = \
                                np.unravel_index(A[sample][channel][i : i + self.kernel, 
                                                                    j : j + self.kernel].argmax(), 
                                                 (self.kernel, self.kernel))
                        Z[sample][channel][i, j] = \
                                A[sample, channel, 
                                  self.pidx[sample][channel][i, j][0] + i,
                                  self.pidx[sample][channel][i, j][1] + j]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        input_width  = output_width + self.kernel - 1
        input_height = output_height + self.kernel - 1

        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))

        #print(dLdZ.shape)
        #print(self.pidx.shape)
        #print(dLdA.shape)

        # Loop through each sample
        for sample in range(batch_size):
            # Loop through each input channel
            for channel in range(in_channels):
                # Loop through width
                for i in range(output_width):
                    # Loop through height
                    for j in range(output_height):
                        dLdA[sample, channel,
                             self.pidx[sample][channel][i, j][0] + i, 
                             self.pidx[sample][channel][i, j][1] + j] += dLdZ[sample][channel][i, j]        

        return dLdA

class MeanPool2d_stride1:

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width  = input_width  - self.kernel + 1
        output_height = input_height - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        # Loop through each sample
        for sample in range(batch_size):
            # Loop through each input channel
            for channel in range(in_channels):
                # Loop through width
                for i in range(output_width):
                    # Loop through height
                    for j in range(output_height):
                        Z[sample][channel][i, j] = np.mean(A[sample][channel][i : i + self.kernel, j : j + self.kernel])

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        input_width  = output_width  + self.kernel - 1
        input_height = output_height + self.kernel - 1

        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))

        # Loop through each sample
        for sample in range(batch_size):
            # Loop through each input channel
            for channel in range(in_channels):
                # Loop through width
                for i in range(output_width):
                    # Loop through height
                    for j in range(output_height):
                        # Loop through filter width
                        for x in range(self.kernel):
                            # Loop through filter height
                            for y in range(self.kernel):
                                dLdA[sample][channel][x + i, y + j] += dLdZ[sample][channel][i, j] / np.pow(self.kernel, 2)

        return dLdA


class MaxPool2d:

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ)

        return dLdA


class MeanPool2d:

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MeanPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ)

        return dLdA
