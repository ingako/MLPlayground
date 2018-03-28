import numpy as np

## This function implements the Viterbi algorithm, to find the most likely
#  sequence of states given some set of observations.
#
## INPUT
#  p is a matrix of transition probabilies for states x;
#  pi is a vector of prior distributions for states x;
#  b is a matrix of emission probabilities;
#  y is a vector of observations.
#
## OUTPUT
#  x is the most likely sequence of states, given the inputs.

def HMM(p, pi, b, y):
    n = len(y)
    m = len(pi)
    
    gamma = np.zeros(shape=(m,n)) # [letter, observation]
    phi = np.zeros(shape=(m,n))
	
    # Initialization 
    # gamma = pi_state * b_i_obs
    # b_i_j = probability of intending to hit key i but hitting key j instead
    for i in range(0, 26):
        gamma[i][0] = pi[i] * b[i][y[0]]
        
    # t = for observations at time t
    for t in range(1, n):
        for k in range(0, 26):
            for j in range(0, 26):
                gamma[k][t] = max(gamma[k][t], gamma[j][t - 1] * p[j][k] * b[k][y[t]])
    
    best = 0
    x = np.zeros(shape=(1, n))
    # Find the final state in the most likely sequence x(n)
    for k in range(0, 26):
        if best <= gamma[k][n - 1]:
            best = gamma[k][n - 1]
            x[0][n - 1] = k
            
    for t in range(n - 1, 0, -1):
        k = x[0][t]
        for j in range(0, 26):
            if phi[k][t] <= gamma[j][t - 1] * p[j][k]:
                phi[k][t] = gamma[j][t - 1] * p[j][k]
                x[0][t - 1] = j
    
    return x[0]
	