Implementation of algorithms used in hidden markov models. 

files: 
 - `backward.py`: an implementation of the backward algorithm that computes the likelihood of an observation given parameters. It is computed recursively backwards from time T to time 0.
 - `baum_welch.py`: an implementation of the baum-welch algorithm used to estimate the parameters that maximize the likelihood of an  observation. 
 - `example.ipynb`: examples on how to use the algorithms. 
 - `forward.py`:  an implementation of the forward algorithm that computes the likelihood of an observation given parameters. It is computed recursively forwards from time 0 to time T.
 - `viterbi.py`: an implementation of the viterbi algorithm that decodes the most likely hidden states that led to a sequence of observations. 
 - `requirements.txt`: contains required python modules to run the algorithms


 To learn more about each algorithm check out my [notes](https://freth251.github.io/digital-garden/maths/Hidden-Markov-Model). 