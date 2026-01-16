import numpy as np
import pandas as pd

from mobility.utils import get_logger

logger = get_logger(__name__)

class MarkovChain:
    '''
    Description
    -----------
    Class dedicated to computing transition probabilities between states and generating predictions based on the computed probabilities.
    Developed specifically for location prediction and movement simulation.
    '''

    def __init__(self, time_gap:int=8, length:int=25, n_sims:int=5):
        '''
        Description
        -----------
        Class dedicated to computing transition probabilities between states and generating predictions based on the computed probabilities.
        Developed specifically for location prediction and movement simulation.

        Parameters
        ----------
        time_gap : int, default=8
            The maximimum amount of time, in hours, that separate one state from the state immediately following it. 
            This is used to prevent computing the transition probability of one state to the next if their is a likelhood that the transition is the result of data quality issues.
            For example, if data collection is sparse on a given day and results in only a single detected staypoint, 
            computing the transition probability of any state to or from that observed event could result inaccurate results

        length : int, default=25
            The number of state transitions to predict when calling `.predict()` or `.fit_predict()`

        n_sims : int, default=5
            The value must be an odd number or a `RuntimeError()` is raised. This number represents the number of simulations that are run when calling `.predict()` or `.fit_predict()`.
            After all simulations complete, the median value for all simulations at a given index are returned as the final prediction. 

        Raises
        ------
        `RuntimeError()`: If argument passed to `n_sims` is NOT an odd integer
        '''
        if n_sims % 2 == 0:
            raise RuntimeError(f"Error: Argument passed for `n_sims` must be odd.")
        
        self.time_gap = time_gap
        self.length = length
        self.n_sims = n_sims 
        self.states = None
        self.matrices = None
        self._is_fitted = False

        logger.debug("MarkovChain successfully initialized.")

    def fit_predict(self, locations:pd.Series, hours:pd.Series, start:int):
        '''
        Description
        -----------
        Public method chaining the `fit()` and `.predict()` methods together.
        Calling this method will first compute the probabilities of a known state transition from one state to a subsequent state
        and then will generate a prediction based on the computed probabilities.

        Parameters
        ----------
        locations : pd.Series
            The semantic labels, ordered by datetime, that will be used to generate the transition probability matrix by 
            determining the likelihood of transitioning from label at index `i` to label at index `i+1`

        hours : pd.Series
            The hours that correspond to each semantic label included in the locations argument. The time delta resulting from the absolute difference between the hour that corresponds to the location at index `i`
            and the hour that corresponds to the location and index `i+1` is used against the argument for `time_gap` passed at initialization. If the time delta is > `time_gap`,
            the that transition is not inlcuded in the probability matrix calculation.

        start : int, default=None
            An integer representing the start of the sequence the user wishes to generate. 
        
        Returns
        ------- 
        sequence : list
            A list of predicted values with the argument passsed for `start` beginning the sequence
        '''
        return self.fit(locations, hours).predict(start)
    
    def fit(self, locations:pd.Series, hours:pd.Series):
        '''
        Description
        -----------
        Public method for computing the probabilities of a known state transition from one state to a subsequent state.

        Parameters
        ----------
        locations : pd.Series
            The semantic labels, ordered by datetime, that will be used to generate the transition probability matrix by 
            determining the likelihood of transitioning from label at index `i` to label at index `i+1`

        hours : pd.Series
            The hours that correspond to each semantic label included in the locations argument. The time delta resulting from the absolute difference between the hour that corresponds to the location at index `i`
            and the hour that corresponds to the location and index `i+1` is used against the argument for `time_gap` passed at initialization. If the time delta is > `time_gap`,
            the that transition is not inlcuded in the probability matrix calculation. 
        
        Returns
        ------- 
        self : MarkovChain
            The model fitted
        '''
        if len(locations) < 2:
            logger.debug("Error, argument for locations does not include enough states to calculate probability matrix.")
            raise ValueError("Error, argument for locations does not include enough states to calculate probability matrix.")
        
        states = locations.unique()
        self.state_range = np.arange(states.min(), states.max()+1).size
        trans_mat = np.zeros((self.state_range, self.state_range), dtype=np.float32)

        for i in range(1, len(locations)):
            time_delta = abs(hours[i-1] - hours[i])
            if time_delta <= self.time_gap:
                origin = locations[i-1]
                dest = locations[i]

                if pd.notna(origin) or pd.notna(dest):
                    trans_mat[origin, dest] += 1

        for i in range(self.state_range):

            row_sum = trans_mat[i, :].sum()
            if row_sum > 0:
                trans_mat[i, :] /= row_sum 
            else:
                trans_mat[i, :] = 1.0 / self.state_range

        self.matrix = trans_mat
        self._is_fitted = True
        return self

    def predict(self, start:int):
        '''
        Description
        -----------
        Public method for simulating a series of sequences based on the argument passed for `n_sims` at object initialization.
        The final prediction is based on the median of all simulations for a given index.

        Parameters
        ----------
        start : int, default=None
            An integer representing the start of the sequence the user wishes to generate. 
        
        Returns
        ------- 
        sequence : list
            A list of predicted values with the argument passsed for `start` beginning the sequence
        '''
        self._fit_check()

        predictions = np.zeros((self.length, self.n_sims))

        for i in range(self.n_sims):
            predictions[:, i] = self._generate_sequence(start)
        
        return np.median(predictions, axis=1)
        

    def get_transition_matrix(self):
        '''
        Description
        -----------
        Public method for access the calculated transition matrix (probabilities of transition from one state to a subsequent state).

        Returns
        -------
        Matrix : 2-d NumPy Array
            Transition matrix
        '''
        self._fit_check()
        return self.matrix

    def _generate_sequence(self, start:int):
        '''
        Description
        -----------
        Private method for generating a sequence based on probability matrix
        calculated during `.fit()`
        
        Parameters
        ----------
        start : int
            An integer representing the start of the sequence the user wishes to generate. 
        
        Returns
        ------- 
        sequence : list
            A list of predicted values ordered beginning with the argument passsed for `start`
        '''
        sequence = [start]
        current_state = start
        for _ in range(self.length - 1):
            probas = self.matrix[current_state, :]
            pred = np.random.choice(
                self.state_range,
                p=probas
            )
            current_state = pred
            sequence.append(pred)

        return np.array(sequence, dtype=int)
    
    def _fit_check(self):
        '''
        Description
        -----------
        Utility function used to raise a `RuntimeError()` whenever a public method is called before `.fit()` or `.fit_predict()`.
        '''
        if not self._is_fitted:
            raise RuntimeError("MarkovChain must be fitted before performing this operation.")