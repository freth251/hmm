
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import sys
def baum_welch(A: np.ndarray, B:np.ndarray, pi:np.ndarray, O:list, steps:int=100, show_plt:bool=True, converge:float=None)->Union[ np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a set of parameters λ=(pi, A, B) and a sequence of observations O
    return an new set of parameters λ'=(pi',A',B') such that P(O|λ')≥P(O|λ). 
    Args:
        A (np.ndarray): A 2D array of size nxn representing the transition probabilities of a Markov Model.
        B (array_like): A 2D array of size nxm representing the emission probabilities of a Hidden Markov Model.  
        pi (np.ndarray): A 2D array of size nx1 representing the stationary distibution. 
        O (list): A list of observations
        steps (int): Number of steps to run the algorithm for, defaults to 100
        show_plt (bool): Show plot of likelihood vs timesteps
        converge (float): Stop training when log likelihood changes by less than this percentage, defaults to None 
    
    Returns: 
        A' (np.ndarray): A 2D array of size nxn representing the transition probabilities of a Markov Model.
        B' (array_like): A 2D array of size nxm representing the emission probabilities of a Hidden Markov Model.  
        pi' (np.ndarray): A 2D array of size nx1 representing the stationary distibution. 
    """

    if show_plt:
        x_data, y_data = [], [] # used for plotting
    last_prob=sys.float_info.max
    for st in range(steps):

        # Calculate temporary variables
        x, prob=get_xi(A, B, pi, O)
        y= get_gamma(x, A.shape[0], len(O))

        # Calculate stationary distribution
        pi= y[0]

        # Calculate transition probability
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                aN=sum(x[t][i][j] for t in range(len(O)-1))
                aD=sum(y[t][i] for t in range(len(O)-1))
                if aD==0:
                    A[i][j]=0
                else:
                    A[i][j]=aN/aD

        # Calculate emission probability
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                yN=sum(y[t][i] for t in range(len(O)) if O[t] == j)
                yD=sum(y[t][i] for t in range(len(O)))
                if yD==0:

                    B[i][j]=0
                else:
                    B[i][j]=yN/yD
        

        log_prob=-np.log(prob)

        if converge is not None: 
            if log_prob!=0:

                diff= 100-last_prob*100/log_prob
                if diff>converge:
                    print(f"Converged at timestep:{st}, current likelihood: {log_prob}, last likelihood: {last_prob}, diff:{diff}%")
                    break
                else:
                    last_prob=log_prob
        
        if show_plt:
            x_data.append(st)
            y_data.append(log_prob)
    
    if show_plt:
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, marker='o', linestyle='-', color='b')

        # Adding titles and labels
        plt.title('Log Likelihood vs. Step')
        plt.xlabel('Step')
        plt.ylabel('Log Likelihood')

        # Display the plot
        plt.grid(True)
        plt.show()

    return A, B, pi




def get_alpha(A: np.ndarray, B:np.ndarray, pi:np.ndarray, O:list)->Union[list[list[float]], float]:
    """
    Given observation O from time 0 to T and parameters λ=(A, B, pi), returns 
    alpha[t][i], the likelihood we end up in state i given all our observation
    up until time t, and p, the likelihood of observation O, using the forward algorithm. 

    Args: 
        A (np.ndarray): A 2D array of size nxn representing the transition probabilities of a Markov Model.
        B (array_like): A 2D array of size nxm representing the emission probabilities of a Hidden Markov Model.  
        pi (np.ndarray): A 2D array of size nx1 representing the stationary distibution. 
        O (list): A list of observations
    Returns: 
        alpha list[list[float]]: A 2D list where each row represents time t and each colums represents state i
        p float: likelihood of observation O, computed using the forward algorithm
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

    return alpha, p

def get_beta(A: np.ndarray, B:np.ndarray, pi:np.ndarray, O:list)->list[list[float]]:
    """
    Given observation O from time 0 to T and parameters λ=(A, B, pi), returns 
    beta[t][i], the likelihood we end up in state i knowing all our observation
    from time t to T, using the backward algorithm. 

    Args: 
        A (np.ndarray): A 2D array of size nxn representing the transition probabilities of a Markov Model.
        B (array_like): A 2D array of size nxm representing the emission probabilities of a Hidden Markov Model.  
        pi (np.ndarray): A 2D array of size nx1 representing the stationary distibution. 
        O (list): A list of observations
    Returns: 
        beta list[list[float]]: A 2D list where each row represents time t and each colums represents state i
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
    
    return beta

def get_xi(A: np.ndarray, B:np.ndarray, pi:np.ndarray, O:list)->list[list[list[float]]]:
    """
    Given observation O and parameters λ=(A, B, pi), returns xi[t][i][j], the probability
    of being in state i at time=(t) then state j at time=(t+1). 

    Args: 
        A (np.ndarray): A 2D array of size nxn representing the transition probabilities of a Markov Model.
        B (array_like): A 2D array of size nxm representing the emission probabilities of a Hidden Markov Model.  
        pi (np.ndarray): A 2D array of size nx1 representing the stationary distibution. 
        O (list): A list of observations
    Returns: 
        xi list[list[list[float]]]: A 2D list where each row represents time t and each colums represents state i
        p float: likelihood of observation O, computed using the forward algorithm
    """
    states=[i for i in range(A.shape[0])]
    steps=len(O)
    x= [[[0.0 for _ in range(len(states))] for _ in range(len(states))] for _ in range(steps)]
    
    alpha, p= get_alpha(A, B, pi, O)
    beta= get_beta(A, B, pi, O)

    for t in range(steps-1): 
        eD=sum(sum(alpha[t][i]*A[j][i]*B[j][O[t+1]]*beta[t+1][j] for j in states) for i in states)
        for i in states: 
            for j in states:
                eN=alpha[t][i]*A[j][i]*B[j][O[t+1]]*beta[t+1][j]
                
                if eD==0:
                    
                    x[t][i][j]=0
                else:
                    x[t][i][j]=eN/eD
    return x, p
 
def get_gamma(x:list[list[list[float]]], n, steps)->list[list[float]]:
    """
    Given observation O and parameters λ=(A, B, pi), returns gamma[t][i], the probability
    of being in state i at time t. 

    Args: 
        A (np.ndarray): A 2D array of size nxn representing the transition probabilities of a Markov Model.
        B (array_like): A 2D array of size nxm representing the emission probabilities of a Hidden Markov Model.  
        pi (np.ndarray): A 2D array of size nx1 representing the stationary distibution. 
        O (list): A list of observations
    Returns: 
        beta list[list[float]]: A 2D list where each row represents time t and each colums represents state i
    """
    states=[i for i in range(n)]

    gamma=[[0.0 for _ in range(len(states))] for _ in range(steps)]

    for t in range(steps):
        for i in states:
            gamma[t][i]= sum(x[t][i][j] for j in states)
            
    return gamma