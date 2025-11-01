import numpy as np
import sys

sys.path.append("mydnn")
from gru_cell import *
from linear import *


class CharacterPredictor(object):
    """CharacterPredictor class.

    This is the neural net that will run one timestep of the input
    You only need to implement the forward method of this class.
    This is to test that your GRU Cell implementation is correct when used as a GRU.

    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CharacterPredictor, self).__init__()

        """
        The network consists of a GRU Cell and a linear layer.
        We refer to the linear layer self.projection in the code 
        because it is just a linear transformation between the hidden state to the output state
        """
        # Check empty params
        if input_dim < 1:
            raise ValueError('input_dim should be bigger than 0')

        if hidden_dim < 1:
            raise ValueError('hidden_dim should be bigger than 0')
        
        if num_classes < 1:
            raise ValueError('num_classes should be bigger than 0')

        self.gru         = GRUCell(input_dim, hidden_dim)
        self.projection  = Linear(hidden_dim, num_classes)
        self.num_classes = num_classes
        self.hidden_dim  = hidden_dim
        
        # Initialize linear weights
        self.projection.W = np.random.uniform(size = (num_classes, hidden_dim))
        #self.projection.b = np.random.uniform(num_classes, 1) #NOTE: b in this case should NOT be initialized!

    def init_rnn_weights(
        self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh
    ):
        # DO NOT MODIFY
        self.gru.init_weights(
            Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh
        )

    # DO NOT MODIFY
    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """CharacterPredictor forward.

        A pass through one time step of the input

        Input
        -----
        x: (feature_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        logits: (num_classes)
            hidden state at current time-step.

        hnext: (hidden_dim)
            hidden state at current time-step.

        """
        # === Error Checking ===

        # Check datatype
        if not type(x) is np.ndarray:
            raise TypeError('x must be numpy array')

        if not type(h) is np.ndarray:
            raise TypeError('h must be numpy array')

        # Check empty array
        if x.size == 0:
            raise ValueError('x cannot be empty')

        if h.size == 0:
            raise ValueError('h cannot be empty')

        # Check size
        if x.shape[0] == 0 or len(x.shape) != 1:
            raise ValueError('x should be 1D array')

        if h.shape[0] == 0 or len(h.shape) != 1:
            raise ValueError('h should be 1D array')

        # === Calculation Starts ===

        hnext = self.gru(x, h)
        logits = self.projection(hnext)

        return np.mean(logits, axis = 0), hnext


def inference(net, inputs):
    """CharacterPredictor inference.

    An instance of the class defined above runs through a sequence of inputs to
    generate the logits for all the timesteps.

    Input
    -----
    net:
        An instance of CharacterPredictor.

    inputs: (seq_len, feature_dim)
            a sequence of inputs of dimensions.

    Returns
    -------
    logits: (seq_len, num_classes)
            one per time step of input..

    """
    # === Error Checking ===

    # Check datatype
    if not type(net) is CharacterPredictor:
        raise TypeError('net must be an object of CharacterPredictor')

    if not type(inputs) is np.ndarray:
        raise TypeError('inputs must be numpy array')
    
    # Check empty array
    if inputs.size == 0:
        raise ValueError('inputs cannot be empty')

    # Check size
    if inputs.shape[0] == 0 or inputs.shape[1] == 0 or len(inputs.shape) != 2:
        raise ValueError('inputs should be 2D array')

    # === Calculation Starts ===
    
    seq_len = inputs.shape[0]
    
    logits = []
    h_init = np.zeros(net.hidden_dim)

    # Loop through each input
    for t in range(seq_len):
        lt, ht = net(inputs[t], h_init if t == 0 else ht)
        logits.append(lt)

    return np.array(logits)
