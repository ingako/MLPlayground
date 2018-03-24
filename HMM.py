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
	
	gamma = np.zeros(shape=(m,n))
	phi = np.zeros(shape=(m,n))
	
	for i in range(0, 26):
		pass
	
	for t in range(1, n):
		for k in range(0, 26):
			for j in range(0, 26):
				pass
			
	best = 0
	x = np.zeros(shape=(1, n))
	# Find the final state in the ost likely sequence x(n)
	for k in range(0, 26):
		if best <= gamma[k][n]:
			best = gamma[k][n]
			x[n] = k
			
	for i in range(n - 1, 1, -1):
		pass
	
	return x
	