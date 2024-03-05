
import numpy as np
def backward(A: np.ndarray, B:np.ndarray, pi:np.ndarray, O:list)-> float:
    """
    Compute the likelihood of an observation using the backward algorithm. 

    Args:
        A (np.ndarray): A 2D array of size nxn representing the transition probabilities of a Markov Model.
        B (array_like): A 2D array of size nxm representing the emission probabilities of a Hidden Markov Model.  
        pi (np.ndarray): A 2D array of size nx1 representing the stationary distibution. 
        O (list): A list of observations
    
    Returns: 
        float: likelihood of observation O
    """

    states=[i for i in range(A.shape[0])]
    steps=len(O)
    beta=[[0.0 for _ in range(len(states))] for _ in range(steps)]

    # Initialization
    for i in states:
        beta[-1][i]= 1
    
    # Recursion
    for t in range(steps-2, -1, -1):
        for i in states:
            beta[t][i]=sum(beta[t+1][j]*A[i][j]*B[j][O[t+1]] for j in states)
    
    # Termination
    p=sum(pi[i] * B[i][O[0]]*beta[0][i] for i in states)
    return p