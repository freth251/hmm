
import numpy as np
def forward(A: np.ndarray, B:np.ndarray, pi:np.ndarray, O:list)-> float:
    """
    Compute the likelihood of an observation using the forward algorithm. 

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
    alpha=[[0.0 for _ in range(len(states))] for _ in range(steps)]

    # Initialization
    for i in states:
        alpha[0][i]= pi[i] * B[i][O[0]]
    
    # Recursion
    for t in range(1, steps):
        for i in states:
            alpha[t][i]=sum(alpha[t-1][j]*A[j][i]*B[i][O[t]] for j in states)
    
    # Termination
    p=sum(alpha[-1][i] for i in states)
    return p