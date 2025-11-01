import numpy as np

#################################################################################
#---------------- CTC Components ------------------------------------------------
# target, matrix, (target_len)                 target sequence
# logits, matrix, (input_len, len(Symbols))     predicted probabilities
# extSymbols, vector, (2*target_len+1)          output from extendign the target with blanks
# skipConnect, vector, (2*target_len+1)         boolean array containing skip connections
# alpha, matrix, (input_len, 2*target_len+1)    Forward probabilities
# beta, matrix, (input_len, 2*target_len+1)     Backward probabilities
# gamma, matrix, (input_len, 2*target_len+1)    Posterior probabilities
#################################################################################
class CTC(object):

    #DO NOT MODIFY this method
    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------

        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.
        Given an output sequence from an RNN/GRU, 
        we want to extend the target sequence with blanks, 
        where blank has been defined in the initialization.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
                    An array with same length as extSymbols to keep track of whether 
                    an extended symbol Sext(j) is allowed to connect directly to Sext(j-2) 
                    (instead of only to Sext(j-1)) or not. 
                    The elements in the array can be True/False or 1/0. 
                    This will be used in the forward and backward algorithms.

        ex: [0,0,0,1,0,0,0,1,0]
        """
        # === Error Checking ===

        # Check datatype
        if not type(target) is np.ndarray:
            raise TypeError('target must be numpy array')

        # Check empty array
        if target.size == 0:
            raise ValueError('target cannot be empty')

        # Check size
        if target.shape[0] == 0 or len(target.shape) != 1:
            raise ValueError('target should be 1D array')

        # === Calculation Starts ===

        extended_symbols = []
        skip_connect = []

        # === Update extSymbols and skip_connect ===
        target_i  = 0
        target_c  = None

        # Loop through (2 * target_len + 1) slots
        for i in range(target.size * 2 + 1):
            # Add blank to every even slot in extended_symbols
            # Add 0 to every even slot in skip_connect
            if i % 2 == 0:
                extended_symbols.append(0)
                skip_connect.append(0)

            # Add original letter to every odd slot
            else:
                extended_symbols.append(int(target[target_i]))
                
                # Check if target_c is the same as last one
                if target_c is not None and target_c != target[target_i]:
                    skip_connect.append(1)
                
                else:
                    skip_connect.append(0)

                target_c  = target[target_i]
                target_i  += 1
        
        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (list, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (list, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """
        # === Error Checking ===

        # Check datatype
        if not type(logits) is np.ndarray:
            raise TypeError('logits must be numpy array')

        if not type(extended_symbols) is list:
            raise TypeError('extended_symbols must be list')

        if not type(skip_connect) is list:
            raise TypeError('skip_connect must be list')

        # Check empty array
        if logits.size == 0:
            raise ValueError('logits cannot be empty')

        if len(extended_symbols) == 0:
            raise ValueError('extended_symbols cannot be empty')

        if len(skip_connect) == 0:
            raise ValueError('skip_connect cannot be empty')

        # Check size
        if logits.shape[0] == 0 or logits.shape[1] == 0 or len(logits.shape) != 2:
            raise ValueError('logits should be 2D array')

        # === Calculation Starts ===

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros((T, S))

        alpha[0, 0] = logits[0, extended_symbols[0]] # First blank
        alpha[0, 1] = logits[0, extended_symbols[1]] # First actual char

        # Loop through all but first t
        for t in range(1, T):
            alpha[t, 0] = alpha[t - 1, 0] * logits[t, extended_symbols[0]]

            # Loop through all but first symbol in extended_symbols
            for i in range(1, S):
                alpha[t, i] = alpha[t - 1, i] + alpha[t - 1, i - 1]

                # If no skip
                if skip_connect[i] == 1:
                    alpha[t, i] += alpha[t - 1, i - 2]

                alpha[t, i] *= logits[t, extended_symbols[i]]
    
        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (list, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (list, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """
        # === Error Checking ===

        # Check datatype
        if not type(logits) is np.ndarray:
            raise TypeError('logits must be numpy array')

        if not type(extended_symbols) is list:
            raise TypeError('extended_symbols must be list')

        if not type(skip_connect) is list:
            raise TypeError('skip_connect must be list')

        # Check empty array
        if logits.size == 0:
            raise ValueError('logits cannot be empty')

        if len(extended_symbols) == 0:
            raise ValueError('extended_symbols cannot be empty')

        if len(skip_connect) == 0:
            raise ValueError('skip_connect cannot be empty')

        # Check size
        if logits.shape[0] == 0 or logits.shape[1] == 0 or len(logits.shape) != 2:
            raise ValueError('logits should be 2D array')

        # === Calculation Starts ===

        S, T = len(extended_symbols), len(logits)
        beta    = np.zeros((T, S))
        betahat = np.zeros((T, S))

        last_S = S - 1
        last_T = T - 1

        betahat[last_T, last_S]     = logits[last_T, extended_symbols[last_S]]      # Last blank
        betahat[last_T, last_S - 1] = logits[last_T, extended_symbols[last_S - 1]]  # Last actual char

        # Loop through all but last t backwards
        for t in range(last_T - 1, -1, -1):
            betahat[t, last_S] = betahat[t + 1, last_S] * logits[t, extended_symbols[last_S]]

            # Loop through all but last symbol in extended_symbols backwards
            for i in range(last_S - 1, -1, -1):
                betahat[t, i] = betahat[t + 1, i] + betahat[t + 1, i + 1]

                # If no skip
                if i <= last_S - 2 and skip_connect[i + 2] == 1:
                    betahat[t, i] += betahat[t + 1, i + 2]

                betahat[t, i] *= logits[t, extended_symbols[i]]
        
        # Loop through all t backwards
        for t in range(last_T, -1, -1):
            # Loop through all symbols in extended_symbols backwards
            for i in range(last_S, -1, -1):
                beta[t, i] = betahat[t, i] / logits[t, extended_symbols[i]]

        return beta

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """
        # === Error Checking ===

        # Check datatype
        if not type(alpha) is np.ndarray:
            raise TypeError('alpha must be numpy array')

        if not type(beta) is np.ndarray:
            raise TypeError('beta must be numpy array')

        # Check empty array
        if alpha.size == 0:
            raise ValueError('alpha cannot be empty')

        if beta.size == 0:
            raise ValueError('beta cannot be empty')

        # Check size
        if alpha.shape[0] == 0 or alpha.shape[1] == 0 or len(alpha.shape) != 2:
            raise ValueError('alpha should be 2D array')

        if beta.shape[0] == 0 or beta.shape[1] == 0 or len(beta.shape) != 2:
            raise ValueError('beta should be 2D array')

        # === Calculation Starts ===

        T, S = alpha.shape

        gamma = np.zeros((alpha.shape))
        sumgamma = np.zeros((T,))

        # Loop through all t
        for t in range(T):
            # Loop through all symbols
            for i in range(S):
                gamma[t, i] = alpha[t, i] * beta[t, i]
                sumgamma[t] += gamma[t, i]

            # Loop through all symbols
            for i in range(S):
                gamma[t, i] = gamma[t, i] / sumgamma[t]
        
        return gamma

#################################################################################
#---------------- CTC Loss Components ------------------------------------------------
# target, matrix, (batch_size, padded_target_len)       target sequence
# logits, matrix, (seqlength, batch_size, len(Symbols)) predicted probabilities
# input_lengths, vector, batch_size,    length of the inputs
# target_lengths, vector, batch_size,   length of the target
# loss, scalar,                         average divergence between posterior probability 
#                                       gamma and the input symbols y_t^r
# dY, matrix, (seqlength, batch_size, len(Symbols)  Derivative of divergence w.r.t.
#                                                   the input symbols at each time
#################################################################################
class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

                Initialize instance variables

        Argument(s)
                -----------
                BLANK (int, optional): blank label index. Default 0.

        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()

        
    # No need to modify
    def __call__(self, logits, target, input_lengths, target_lengths):
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

                Computes the CTC Loss by calculating forward, backward, and
                posterior proabilites, and then calculating the avg. loss between
                targets and predicted log probabilities

                The loss is average loss.  

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
                        log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """
        # === Error Checking ===

        # Check datatype
        if not type(logits) is np.ndarray:
            raise TypeError('logits must be numpy array')

        if not type(target) is np.ndarray:
            raise TypeError('target must be numpy array')

        if not type(input_lengths) is np.ndarray:
            raise TypeError('input_lengths must be numpy array')

        if not type(target_lengths) is np.ndarray:
            raise TypeError('target_lengths must be numpy array')

        # Check empty array
        if logits.size == 0:
            raise ValueError('logits cannot be empty')

        if target.size == 0:
            raise ValueError('target cannot be empty')

        if input_lengths.size == 0:
            raise ValueError('input_lengths cannot be empty')

        if target_lengths.size == 0:
            raise ValueError('target_lengths cannot be empty')

        # Check size
        if logits.shape[0] == 0 or logits.shape[1] == 0 or logits.shape[2] == 0 or len(logits.shape) != 3:
            raise ValueError('logits should be 3D array')

        if target.shape[0] == 0 or target.shape[1] == 0 or len(target.shape) != 2:
            raise ValueError('target should be 2D array')

        if input_lengths.shape[0] == 0 or len(input_lengths.shape) != 1:
            raise ValueError('input_lengths should be 1D array')

        if target_lengths.shape[0] == 0 or len(target_lengths.shape) != 1:
            raise ValueError('target_lengths should be 1D array')

        # === Calculation Starts ===

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []
        self.skip_connect     = []

        # Loop through all data in batch
        for b in range(B):
            # 1. Truncate input and target
            curr_logits = logits[:input_lengths[b], b, :]
            curr_target = target[b, :target_lengths[b]]

            # 2. Extend target with blanks
            curr_extended_symbols, curr_skip_connect = self.ctc.extend_target_with_blank(curr_target)
            self.extended_symbols.append(curr_extended_symbols)
            self.skip_connect.append(curr_skip_connect)
           
            # 3. Calculate posterior
            curr_gamma = self.ctc.get_posterior_probs(
                self.ctc.get_forward_probs(curr_logits, curr_extended_symbols, curr_skip_connect),
                self.ctc.get_backward_probs(curr_logits, curr_extended_symbols, curr_skip_connect)
            )
            self.gammas.append(curr_gamma)

            # 4. Calculate loss
            T, S = curr_gamma.shape

            # Loop through all t
            for t in range(T):
                # Loop through all symbols
                for s in range(S):
                    total_loss[b] -= curr_gamma[t, s] * np.log(curr_logits[t, curr_extended_symbols[s]])

        return np.mean(total_loss)

    def backward(self):
        """

                CTC loss backard

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.zeros_like(self.logits)

        # Loop through all data in batch
        for b in range(B):
            curr_logits = self.logits[:self.input_lengths[b], b, :]

            # Loop through all t
            for t in range(self.input_lengths[b]):
                # Loop through all symbols
                for i, s in enumerate(self.extended_symbols[b]):
                    dY[t, b, s] -= self.gammas[b][t][i] / curr_logits[t, s]

        return dY
