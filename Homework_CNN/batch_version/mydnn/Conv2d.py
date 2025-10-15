import numpy as np
from resampling import *

################### Conv2d_stride1 and Conv2d Class Components ###################################
# kernel size:  K;  type scalar;    kernel size
# stride:           type: scalar;   downsampling factor
# ------------------------------------------------------------------------------------
# A:    type: Matrix of N x C_in x H_in x W_in;     data input for convolution
# Z:    type: Matrix of N x C_out x H_out x W_out;  features after conv2d with stride 1
# ------------------------------------------------------------------------------------
# W:    type: Matrix of C_out x C_in X K X K;   weight parameters, i.e. kernels
# b:    type: Matrix of C_out x 1;              bias parameters
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x C_out x H_out x W_out;  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x H_in x W_in;     how changes in inputs affect loss
# dLdW: type: Matrix of C_out x C_in X K X K;       how changes in weights affect loss
# dLdb: type: Matrix of C_out x 1;                  how changes in bias affect loss
######################################################################################

class Conv2d_stride1:
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
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size)
            )
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape

        output_height = input_height - self.kernel_size + 1
        output_width  = input_width  - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))

        # Loop through Z width
        for i in range(output_width):
            # Loop through Z height
            for j in range(output_height):
                # Loop through output maps
                for m in range(self.out_channels):
                    # Loop through input maps
                    for n in range(self.in_channels):
                        # Loop through kernel width
                        for ik in range(self.kernel_size):
                            # Loop through kernel height
                            for jk in range(self.kernel_size):
                                Z[:, m, i, j] += np.dot(self.W[m, n, ik, jk],
                                                        A[:, n, i + ik, j + jk])

                    Z[:, m, i, j] += self.b[m] # Add bias

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        #NOTE: originally "batch_size, in_channels, input_height, input_width = self.A.shape"
        #      but I think it should be about dLdZ
        batch_size, out_channels, output_height, output_width = dLdZ.shape

        # ------ Pad dLdZ (NOTE: May not be useful) ------
        #dLdZ_pad = np.zeros((batch_size, out_channels, 
        #                     (output_height + 2 * (self.kernel_size - 1)),
        #                     (output_width  + 2 * (self.kernel_size - 1))
        #                    ))

        # Loop through sample
        #for sample in range(batch_size):
        #    # Loop through channel
        #    for channel in range(out_channels):
        #        dLdZ_pad[sample, channel, 
        #                 self.kernel_size : self.kernel_size + output_height - 1,
        #                 self.kernel_size : self.kernel_size + output_width  - 1
        #                ] = dLdZ[sample, channel, :, :]
        
        # ------Perform back propagation ------
        dLdA = np.zeros(self.A.shape)
        _, in_channels, _, _ = self.A.shape

        # Loop through output width
        for i in range(output_width - 1, -1, -1):
            # Loop through output height
            for j in range(output_height - 1, -1, -1):
                # Loop through output maps
                for m in range(out_channels - 1, -1, -1):
                    # Loop through input maps
                    for n in range(in_channels - 1, -1, -1):
                        # Loop through kernel width
                        for ik in range(self.kernel_size - 1, -1, -1):
                            # Loop through kernel height
                            for jk in range(self.kernel_size - 1, -1, -1):
                                dLdA[:, n, i + ik, j + jk] += np.dot(self.W[m, n, ik, jk],
                                                                     dLdZ[:, m, i, j])

                                self.dLdW[m, n, ik, jk] += np.dot(self.A[:, n, i + ik, j + jk],
                                                                  dLdZ[:, m, i, j])
        
                    self.dLdb[m] = np.sum(dLdZ[:, m, :, :])
                
        return dLdA


class Conv2d:
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

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Save input height and width for unpadding
        self.batch_size, self.in_channels, self.in_height, self.in_width = A.shape
        A_pad = []

        # Pad the input appropriately using np.pad() function
        if self.pad != 0:
            # Loop through sample
            for sample in range(self.batch_size):
                _ = []

                # Loop through channel
                for channel in range(self.in_channels):
                    _.append(np.pad(A[sample, channel], pad_width = self.pad))

                A_pad.append(_)

        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(np.array(A_pad) if self.pad != 0 else A)

        # downsample
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Call downsample1d backward
        dLdZ = self.downsample2d.backward(dLdZ)
        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)
        # Unpad the gradient
        dLdA_unpad = []

        # If padding
        if self.pad != 0:
            # Loop through sample
            for sample in range(self.batch_size):
                _ = []

                # Loop through channel
                for channel in range(self.in_channels):
                    _.append(dLdA[sample, channel, 
                                  self.pad : self.pad + self.in_height,
                                  self.pad : self.pad + self.in_width])

                dLdA_unpad.append(_)

        return np.array(dLdA_unpad) if self.pad != 0 else dLdA
