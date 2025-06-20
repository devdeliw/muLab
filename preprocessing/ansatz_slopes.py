# run from mulab/ 

import os
import pickle 

def display(): 
    fname = "./assets/red_clump_cuts.pkl" 
    if not os.path.exists(fname): 
        print("path not found.")

    with open(fname, "rb") as f:
        slopes = pickle.load(f)

    print(slopes)

if __name__ == "__main__": 
    display()



