import numpy as np
from mydnn.activation import *

#################################################################################
# input_size: H_in, scalar, the number of expected features in the input x
# hidden_size: H_out, scalar, the number of features in the hidden state
# -------------------------------------------------------------------------------
# x: x_t,               maxtrix, N x H_in, input at current time step
# h_prev_t: h_{t-1,l},  maxtrix, N x H_out, previous time step hidden state of current layer
# h_t: h_{t,l},         maxtrix, N x H_out, current time step hidden state of current layer
# -------------------------------------------------------------------------------
# W_ih: maxtrix, H_out x H_in,  weight between input and hidden
# b_ih: vector,  H_out,         bias between input and hidden
# W_hh: maxtrix, H_out x H_out, weight between previous hidden and current hidden
# b_hh: vector,  H_out,         weight between previous hidden and current hidden
# ----USED in BACKWARD calculation ----------------------------------------------
# delta: maxtrix,   N x H_out,      gradient w.r.t. current hidden layer
# dx: matrix,       N x H_in,       gradient w.r.t. input layer
# -------------------------------------------------------------------------------
# dh_prev_t: maxtrix, N x H_out,    gradient w.r.t. hidden state at previous time step
# dW_ih: maxtrix,   H_out x H_in,   gradient of weight between input and hidden
# db_ih: vector,  H_out,            gradient of bias between input and hidden
# dW_hh: maxtrix, H_out x H_out,    gradient of weight between previous hidden and current hidden
# db_hh: vector,  H_out,             gradient of bias between previous hidden and current hidden
#################################################################################

class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):
        
        # Check empty array
        if input_size < 1:
            raise ValueError('input_size should be bigger than 0')

        if hidden_size < 1:
            raise ValueError('hidden_size should be bigger than 0')

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # Weight definitions
        # ---------------------------start
        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)
        # ---------------------------end

        # Gradient definitions
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros((h))
        self.db_hh = np.zeros((h))
        
    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    # DO NOT change this method
    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    # DO NOT change this method
    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        RNN Cell forward (single time step).

        Input 
        -----
        x: (batch_size, input_size)
            input at the current time step

        h_prev_t: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns
        -------
        h_t: (batch_size, hidden_size)
            hidden state at the current time step and current layer
        """

        """
        ht = tanh(W_ih x_t + b_ih + W_hh h_t−1 + b_hh) 
        """
        # === Error Checking ===

        # Check datatype
        if not type(x) is np.ndarray:
            raise TypeError('x must be numpy array')

        if not type(h_prev_t) is np.ndarray:
            raise TypeError('h_prev_t must be numpy array')

        # Check empty array
        if x.size == 0:
            raise ValueError('x cannot be empty')

        if h_prev_t.size == 0:
            raise ValueError('h_prev_t cannot be empty')

        # Check size
        if x.shape[0] == 0 or x.shape[1] == 0 or len(x.shape) != 2:
            raise ValueError('x should be 2D array')

        if h_prev_t.shape == 0 or h_prev_t.shape[1] == 0 or len(h_prev_t.shape) != 2:
            raise ValueError('h_prev_t should be 2D array ')

        # === Calculation Starts ===

        # Forward calculation
        h_t = self.activation.forward(
                np.dot(x, self.W_ih.T) + np.tile(self.b_ih, (x.shape[0], 1)) +
                np.dot(h_prev_t, self.W_hh.T) + np.tile(self.b_hh, (h_prev_t.shape[0], 1))
              )

        return h_t

    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        """
        RNN Cell backward (single time step).

        Input 
        -----
        delta: (batch_size, hidden_size)
                Gradient w.r.t the current hidden layer

        h_t: (batch_size, hidden_size)
            Hidden state of the current time step and the current layer

        h_prev_l: (batch_size, input_size)
                    Hidden state at the current time step and previous layer

        h_prev_t: (batch_size, hidden_size)
                    Hidden state at previous time step and current layer

        Returns
        -------
        dx: (batch_size, input_size)
            Derivative w.r.t. the current time step and previous layer

        dh_prev_t: (batch_size, hidden_size)
            Derivative w.r.t. the previous time step and current layer

        """
        # === Error Checking ===
        
        # Check datatype
        if not type(delta) is np.ndarray:
            raise TypeError('delta must be numpy array')

        if not type(h_t) is np.ndarray:
            raise TypeError('h_t must be numpy array')
        
        if not type(h_prev_l) is np.ndarray:
            raise TypeError('h_prev_l must be numpy array')

        if not type(h_prev_t) is np.ndarray:
            raise TypeError('h_prev_t must be numpy array')

        # Check empty array
        if delta.size == 0:
            raise ValueError('delta cannot be empty')

        if h_t.size == 0:
            raise ValueError('h_t cannot be empty')

        if h_prev_l.size == 0:
            raise ValueError('h_prev_l cannot be empty')

        if h_prev_t.size == 0:
            raise ValueError('h_prev_t cannot be empty')

        # Check size
        if delta.shape[0] == 0 or delta.shape[1] == 0 or len(delta.shape) != 2:
            raise ValueError('delta should be 2D array')

        if h_t.shape == 0 or h_t.shape[1] == 0 or len(h_t.shape) != 2:
            raise ValueError('h_t should be 2D array ')

        if h_prev_l.shape[0] == 0 or h_prev_l.shape[1] == 0 or len(h_prev_l.shape) != 2:
            raise ValueError('h_prev_l should be 2D array')

        if h_prev_t.shape == 0 or h_prev_t.shape[1] == 0 or len(h_prev_t.shape) != 2:
            raise ValueError('h_prev_t should be 2D array ')

        # === Calculation starts ===
        
        batch_size = delta.shape[0]
   
        # Add necessary code to calculate dz 
        dz = self.activation.backward(delta, h_t)
        
        # Add necessary code to compute the averaged gradients (per batch) of the weights and biases
        self.dW_ih += np.dot(dz.T, h_prev_l) / batch_size
        self.dW_hh += np.dot(dz.T, h_prev_t) / batch_size
        self.db_ih += np.mean(dz, axis = 0)
        self.db_hh += np.mean(dz, axis = 0)
        

        # Add necessary code to compute dx, dh_prev_t
        dx = np.dot(dz, self.W_ih)
        dh_prev_t = np.dot(dz, self.W_hh)
        
        return dx, dh_prev_t
