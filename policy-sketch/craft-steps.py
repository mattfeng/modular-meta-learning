import numpy as np
import itertools
import random

def inbounds(loc, R, C):
    return 0 <= loc[0] < R and 0 <= loc[1] < C

def neighbors(r, c, R, C):
    cur = np.array([r, c])
    offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    offsets = map(np.array, offsets)
    return [(cur + offset) for offset in offsets 
            if inbounds(cur + offset, R, C)]

def main():
    R, C = 10, 10
    gr = 9
    gc = 9

    steps = 0
    T = 50000
    for t in range(T):
        r = 4
        c = 4
        print(f"Trial: {t}")
        while (r, c) != (gr, gc):
            r, c = random.choice(neighbors(r, c, R, C))
            steps += 1
    
    print(f"Average steps taken: {steps / T}")


if __name__ == "__main__":
    main()