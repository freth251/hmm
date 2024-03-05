
import numpy as np
from collections import defaultdict
def viterbi(A: np.ndarray, B:np.ndarray, pi:np.ndarray, O:list)-> list:
    """
    Compute the likeliest sequence of hidden states given an observation using the viterbi algorithm. 

    Args:
        A (np.ndarray): A 2D array of size nxn representing the transition probabilities of a Markov Model.
        B (array_like): A 2D array of size nxm representing the emission probabilities of a Hidden Markov Model.  
        pi (np.ndarray): A 2D array of size nx1 representing the stationary distibution. 
        O (list): A list of observations
    
    Returns: 
        list: sequence of hidden states
    """

    states=[i for i in range(A.shape[0])]
    steps= len(O)

    # sequence[i][t] represents the likeliest sequence of states at time t and state i given observations till time t-1
    sequence=[[row if col == 0 else -1 for col in range(steps)] for row in range(len(states))]

    # Initialization
    alpha_t=[]

    a=[]
    for s in states:
        alpha_1= pi[s] * B[s][O[0]]
        a.append(alpha_1)
    alpha_t.append(a)


    # Recursion
    for t in range(1, steps):
        a=[]
        for i in states:
            f=0
            maxV=0
            for j in states: 
                g=alpha_t[t-1][j]*A[j][i]*B[i][O[t]]
                if g>f:
                    f=g
                    maxV=j
            sequence[i][t]=maxV
            a.append(f)
        alpha_t.append(a)

    # Termination
    c=alpha_t[-1].index(max(alpha_t[-1]))
    sequence[c][-1]=c
    return sequence[c]