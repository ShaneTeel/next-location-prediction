import numpy as np
import pandas as pd

from mobility.utils import get_logger

logger = get_logger(__name__)

class MarkovChain:

    def __init__(self, states:pd.Series):
       self.states = states.unique()
       self.num_states = len(self.states)
       self.matrix = self._initialize_transition_matrix(self.num_states, states)

       logger.debug("MarkovChain successfully initialized.")
    
    def _initialize_transition_matrix(self, num_states:int, locations:pd.Series):
        dest = locations.shift(-1, fill_value=0).astype(int)

        trans_mat = np.zeros((num_states, num_states), dtype=np.float32)
        
        for i in range(len(locations)):
            a = locations[i]
            b = dest[i]
            if pd.notna(a) or pd.notna(b):
                trans_mat[a, b] += 1

        for i in range(num_states):
            row_sum = trans_mat[i, :].sum()
            if row_sum > 0:
                trans_mat[i, :] /= row_sum

        return trans_mat

    def generate_sequence(self, start:int, length:int):
        '''
        Description
        -----------
        Public method for generating a sequence based on probability matrix
        calculated during initialization
        
        Parameters
        ----------
        start : int
            An integer representing the start of the sequence the user wishes to generate

        length : int 
            The length of the sequence the user wishes to generate
        
        Returns
        ------- 
        sequence : list
            A list of predicted values ordered beginning with the argument passsed for `start`
        '''
        sequence = [start]

        for _ in range(length - 1):
            probas = self.matrix[start, :]
            pred = np.random.choice(
                self.states,
                p=probas
            )
            sequence.append(pred)
        return sequence