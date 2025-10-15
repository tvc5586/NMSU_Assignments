# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

################### Conv1d_stride1 and Conv1d Class Components ###################################
# kernel size:  K;      type scalar;        kernel size
# stride:               type: scalar;       equivalent to downsampling factor
# ------------------------------------------------------------------------------------
# A:    type: Matrix of N x C_in x W_in;    data input for convolution
# Z:    type: Matrix of N x C_out x W_out;  features after conv1d with stride
# ------------------------------------------------------------------------------------
# W:    type: Matrix of C_out x C_in X K;   weight parameters, i.e. kernels
# b:    type: Matrix of C_out x 1;          bias parameters
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x C_out x W_out;  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x W_in;    how changes in inputs affect loss
# dLdW: type: Matrix of C_out x C_in X K;   gradient of Loss w.r.t. weights
# dLdb: type: Matrix of C_out x 1;          gradient of Loss w.r.t. bias
###################################################################################
class Conv1d_stride1:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, in_channels, input_size = A.shape

        output_size = input_size - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_size))

        # Loop through sample
        for sample in range(batch_size):
            # Loop through Z size 
            for i in range(output_size):
                # Loop through output maps
                for m in range(self.out_channels):
                    # Loop through input maps
                    for n in range(self.in_channels):
                        # Loop through kernel width
                        for ik in range(self.kernel_size):
                            Z[sample, m, i] += np.dot(self.W[m, n, ik],
                                                      A[sample, n, i + ik])

                    Z[sample, m, i] += self.b[m] # Add bias

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape
        
        dLdA = np.zeros(self.A.shape)
        _, in_channels, _ = self.A.shape

        # Loop through sample
        for sample in range(batch_size):
            # Loop through output size
            for i in range(output_size - 1, -1, -1):
                # Loop through output maps
                for m in range(out_channels - 1, -1, -1):
                    # Loop through input maps
                    for n in range(in_channels - 1, -1, -1):
                        # Loop through kernel width
                        for ik in range(self.kernel_size - 1, -1, -1):
                            dLdA[sample, n, i + ik] += np.dot(self.W[m, n, ik],
                                                                  dLdZ[sample, m, i])

                            self.dLdW[m, n, ik] += np.dot(self.A[sample, n, i + ik],
                                                          dLdZ[sample, m, i])
                            self.dLdb[m] = np.sum(dLdZ[:, m, :])

        return dLdA


class Conv1d:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # Save input size for unpadding
        batch_size, self.in_channels, self.in_size = A.shape
        A_pad = []

        # Calculate Z
        # Line 1: Pad with zeros
        # If padding
        if self.pad != 0:
            # Loop through sample
            for sample in range(batch_size):
                _ = []

                # Loop through channel
                for channel in range(self.in_channels):
                    _.append(np.pad(A[sample][channel], pad_width = self.pad))

                A_pad.append(_)

        # Line 2: Conv1d forward
        Z = self.conv1d_stride1.forward(np.array(A_pad) if self.pad != 0 else A)
        # Line 3: Downsample1d forward
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, _, _ = dLdZ.shape

        # Calculate dLdA
        # Line 1: Downsample1d backward
        dLdZ = self.downsample1d.backward(dLdZ)
        # Line 2: Conv1d backward
        dLdA = self.conv1d_stride1.backward(dLdZ)
        # Line 3: Unpad
        # If padding
        dLdA_unpad = []

        if self.pad != 0:
            # Loop through sample
            for sample in range(batch_size):
                _ = []

                # Loop through channel
                for channel in range(self.in_channels):
                    _.append(dLdA[sample][channel][self.pad : self.pad + self.in_size])

                dLdA_unpad.append(_)

        return np.array(dLdA_unpad) if self.pad != 0 else dLdA
