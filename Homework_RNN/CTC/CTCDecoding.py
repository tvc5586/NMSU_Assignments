import numpy as np


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            A list of symbols that can be predicted, except for the blank symbol.

        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """
        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            The probability distribution over all symbols including the blank 
            symbol at each time step. The probability of blank for all time steps 
            is the first row of y_probs (index 0).
            Be careful with the batch size in all test cases. 
            If it is not 1, please make sure to incorporate batch_size. 

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path, temp_path = [], []
        blank = 0
        path_prob = 1

        #add necessary code
        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
    
        S, T, B = y_probs.shape

        # Loop through all data
        for b in range(B):
            # Loop through all t
            for t in range(T):
                curr_max_prob = 0
                curr_symbol   = 0

                # Loop through all symbols
                for s in range(S):
                    # Compare max prob symbol with current symbol
                    if curr_max_prob < y_probs[s, t, b]:
                        curr_max_prob = y_probs[s, t, b]
                        curr_symbol = 0 if s == 0 else self.symbol_set[s - 1]

                path_prob *= curr_max_prob
                temp_path.append(curr_symbol)

        # Loop through all chars in path
        for i in range(len(temp_path)):
            # Find out duplicates that are next to each other
            if i != 0 and temp_path[i] == temp_path[i - 1]:
                continue

            decoded_path.append(temp_path[i])

        # Remove blank
        decoded_path = [s for s in decoded_path if s != blank]

        return "".join(decoded_path), path_prob

