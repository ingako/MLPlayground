import numpy as np

## This function constructs tranisition matricies for lowercase characters.
#  It is assumed that the file 'filename' only contains lowercase characters
#  and whitespace.
## INPUT
#  filename is the file containing the text from which we wish to develop a
#  Markov process.
#
## OUTPUT
#  p is a 26 x 26 matrix containing the probabilities of transition from a
#  state to another state, based on the frequencies observed in the text.
#  prior is a vector of prior probabilities based on how often each character
#  appears in the text
#
## Read the file into a string called text
def constructTransitions(filename):
    p = np.zeros(shape=(26, 26)) # transition matrix
    prior = np.zeros(shape=(26, 1)) # prior vector
    
    with open(filename) as f:
        for line in f:
            for word in line.split():
                for i, c in enumerate(word):
                    to_c = ord(c) - ord('a')
                    prior[to_c] += 1
                    
                    if i == 0: 
                        continue
                    from_c = ord(word[i - 1]) - ord('a')
                    p[from_c][to_c] += 1

    for i in range(0, 26): 
        p[i] /= sum(p[i])
    prior /= sum(prior)
    
    return (p, prior)