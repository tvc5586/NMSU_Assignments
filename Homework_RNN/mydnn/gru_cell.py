import numpy as np
from mydnn.activation import *

# 
# This class is going to replicate a portion of the torch.nn.GRUCell interface
# 
#################################################################################
# input_size: H_in, scalar, the number of expected features in the input x
# hidden_size: H_out, scalar, the number of features in the hidden state
# -------------------------------------------------------------------------------
# x: x_t,               vector, H_in, observation at the current time step
# h_prev_t: h_{t-1},    vector, H_out, hidden state at the previous time step
# -------------------------------------------------------------------------------
# W_rx: maxtrix, H_out x H_in,  weight matrix for input (for reset gate)
# W_zx: maxtrix, H_out x H_in,  weight matrix for input (for update gate)
# W_nx: maxtrix, H_out x H_in,  weight matrix for input (for candidate hidden state)
# W_rh: maxtrix, H_out x H_out,  weight matrix for hidden state (for reset gate)
# W_zh: maxtrix, H_out x H_out,  weight matrix for hidden state (for update gate)
# W_nh: maxtrix, H_out x H_out,  weight matrix for hidden state (for candidate hidden state)
# -------------------------------------------------------------------------------
# b_rx: vector,  H_out,         bias for input (for reset gate)
# b_zx: vector,  H_out,         bias for input (for update gate)
# b_nx: vector,  H_out,         bias for input (for candidate hidden state)
# b_rh: vector,  H_out,         bias for hidden state (for reset gate)
# b_zh: vector,  H_out,         bias for hidden state (for update gate)
# b_nh: vector,  H_out,         bias for hidden state (for candidate hidden state)
# -------------------------------------------------------------------------------
# ----USED in BACKWARD calculation ----------------------------------------------
# delta: vector,   H_out,      gradient of loss w.r.t. h_t
# dx: vector,       H_in,      gradient of loss w.r.t. x_t
# dh_prev_t: vector, H_out,    gradient of loss w.r.t. h_{t-1}
#
# dW_rx: maxtrix, H_out x H_in,  gradient of loss w.r.t. W_rx
# dW_zx: maxtrix, H_out x H_in,  gradient of loss w.r.t. W_zx
# dW_nx: maxtrix, H_out x H_in,  gradient of loss w.r.t. W_nx
# dW_rh: maxtrix, H_out x H_out, gradient of loss w.r.t. W_rh
# dW_zh: maxtrix, H_out x H_out, gradient of loss w.r.t. W_zh
# dW_nh: maxtrix, H_out x H_out, gradient of loss w.r.t. W_nh
#
# db_rx: vector,  H_out,         gradient of loss w.r.t. b_rx
# db_zx: vector,  H_out,         gradient of loss w.r.t. b_zx
# db_nx: vector,  H_out,         gradient of loss w.r.t. b_nx
# db_rh: vector,  H_out,         gradient of loss w.r.t. b_rh
# db_zh: vector,  H_out,         gradient of loss w.r.t. b_zh
# db_nh: vector,  H_out,         gradient of loss w.r.t. b_nh
#################################################################################
class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        
        # Check empty array
        if input_size < 1:
            raise ValueError('input_size should be bigger than 0')

        if hidden_size < 1:
            raise ValueError('hidden_size should be bigger than 0')
       
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d

        self.x_t = 0
        self.h_t = 0
        self.hidden = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    # DO NOT change this method
    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.
        In forward, we calculate h_t. 

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Attributes
        -----
            Forward stores variables x, hidden, r, z, and n. 

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

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
        if x.shape[0] == 0 or len(x.shape) != 1:
            raise ValueError('x should be 1D array')

        if h_prev_t.shape == 0 or len(h_prev_t.shape) != 1:
            raise ValueError('h_prev_t should be 1D array')
        
        # === Calculation Starts ===
        
        self.x_t = x
        self.hidden = h_prev_t

        self.r = self.r_act.forward(np.dot(self.Wrx, x) + self.brx + 
                                    np.dot(self.Wrh, h_prev_t) + self.brh)
        self.z = self.z_act.forward(np.dot(self.Wzx, x) + self.bzx + 
                                    np.dot(self.Wzh, h_prev_t) + self.bzh)
        self.n = self.h_act.forward(np.dot(self.Wnx, x) + self.bnx + 
                                    np.multiply(self.r, (np.dot(
                                                self.Wnh, h_prev_t) + self.bnh)))

        self.h_t = (np.multiply(1 - self.z, self.n) +
                    np.multiply(self.z, h_prev_t))

        return self.h_t

    def backward(self, delta):
        """GRU cell backward.
    
        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # === Error Checking ===
        
        # Check datatype
        if not type(delta) is np.ndarray:
            raise TypeError('delta must be numpy array')

        # Check empty array
        if delta.size == 0:
            raise ValueError('delta cannot be empty')

        # Check size
        if delta.shape[0] == 0 or len(delta.shape) != 1:
            raise ValueError('delta should be 1D array')

        # === Calculation Starts ===
        
        # derivative related to h_t
        dLdn = delta * (1 - self.z)
        dLdz = delta * (self.hidden - self.n)
        dLdh = delta * self.z

        # derivative related to z
        dzdza = self.z_act.backward(dLdz)
        dzadx = np.dot(self.Wzx.T, dzdza)
        dzadh = np.dot(self.Wzh.T, dzdza)
        
        # derivative related to n
        dndna  = self.h_act.backward(dLdn, self.n)
        dnadx  = np.dot(self.Wnx.T, dndna)
        dnadr  = dndna * (np.dot(self.Wnh, self.hidden) + self.bnh)
        dnadhw = dndna * self.r
        
        # derivative related to r
        drdra = self.r_act.backward(dnadr)
        dradx = np.dot(self.Wrx.T, drdra)
        dradh = np.dot(self.Wrh.T, drdra)
        
        # derivative related to h
        dhwdh  = np.dot(self.Wnh.T, dnadhw)
        
        # update weight gradients
        self.dWrx += np.dot(drdra.reshape(-1, 1), self.x_t.reshape(1, -1))
        self.dWzx += np.dot(dzdza.reshape(-1, 1), self.x_t.reshape(1, -1))
        self.dWnx += np.dot(dndna.reshape(-1, 1), self.x_t.reshape(1, -1))
        self.dWrh += np.dot(drdra.reshape(-1, 1), self.hidden.reshape(1, -1))
        self.dWzh += np.dot(dzdza.reshape(-1, 1), self.hidden.reshape(1, -1))
        self.dWnh += np.dot(dnadhw.reshape(-1, 1), self.hidden.reshape(1, -1))
      
        # update bias gradients
        self.dbrx += drdra
        self.dbzx += dzdza
        self.dbnx += dndna
        self.dbrh += drdra
        self.dbzh += dzdza
        self.dbnh += dndna * self.r

        # calculate dx and dh_prev_t
        dx = dzadx + dnadx + dradx
        dh_prev_t = dLdh + dzadh + dradh + dhwdh

        return dx, dh_prev_t
