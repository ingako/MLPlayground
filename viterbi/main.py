import sys
import numpy as np
import constructEmissions as ce
import constructTransitions as ct
import HMM as hmm

# The text messages you have received.
input=['cljlx ypi ktxwf a pwfi psti vgicien aabdwucg vpd me and vtiex voe zoicw',
	   'qe qzby yii tl gp tp yhr cpozwdt fwstqurzby',
	   'qee ypi xfjvkjv ygetw ib ulur vae',
	   'wgrrr zrw uiu',
	   'hpq fzr qee ypi vrpm grfw',
	   'qe zfr xtztvkmh',
	   'wgzf tjmr will uiu xjoq jp ywfw'];

# The probability of hitting the intended key.
pr_correct = 0.5;

# An adjacency matrix set to 1 if the ith letter in the alphabet is next to
# the jth letter in the alphabet on the keyboard.
adj=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,1],
     [0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
     [0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0],
     [0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0],
     [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0],
     [0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0],
     [0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0],
     [0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0],
     [0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
     [0,0,0,0,0,0,0,1,1,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
     [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
     [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1],
     [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
     [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
     [0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
     [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
     [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0]];

def main(argv):
    # Call a function to construct the emission probabilities of hitting a key
    # given you tried to hit a (potentially) different key.
    b = ce.constructEmissions(pr_correct,adj);
    
    # Call a function to construct transmission probabilities and a prior distribution
    # from the King James Bible.
    (p, prior) = ct.constructTransitions('bible.txt')
    
    # Run the Viterbi algorithm on each word of the messages to determine the
    # most likely sequence of characters.
    for t in range(0, len(input)):
        s_in = input[t].split()
        output = ""
        
        for i, word in enumerate(s_in):
            y = np.zeros(shape=(1,len(word)))
            
            for j in range(0, len(word)):
                y[0][j] = ord(word[j]) - ord('a')
                
            # perform the Viterbi algorithm
            x = hmm.HMM(p, prior, b, y[0])
            
            for j in range(0, len(x)):
                output += chr(int(x[j]) + ord('a'))

            if i != len(s_in) - 1:
                output += ' '
        
        print(input[t])
        print(output)
        print()

if __name__ == "__main__":
    main(sys.argv)
