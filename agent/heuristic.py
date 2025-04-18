import numpy as np
from pyparsing import empty

WEIGHT1 = [[4**6, 4**5, 4**3, 4**2],[4**5, 4**4,4**3,4**2],[4**4, 4**3, 4**2, 4**1],[4**3, 4**2, 4**1, 4**0]]
WEIGHT2 = [[4**20, 4**16, 4**14, 4**12],[4**7, 4**8,4**9,4**10],[4**6, 4**5, 4**4, 4**3],[4**3, 4**2, 4**1, 4**0]]

def exp_heuristic(state):
    return np.sum([[WEIGHT1[i][j] * state[i][j] for j in range(4)] for i in range(4)] )
def exp_heuristic2(state):
    heuristic = np.sum([[WEIGHT2[i][j] * state[i][j] for j in range(4)] for i in range(4)] )
    empty_cell_weight = 10
    empty_cell_num = 0
    for i in range(4):
        for j in range(4):
            if state[i][j] == 0:
                empty_cell_num += 1
    heuristic += empty_cell_weight*empty_cell_num**3
    return heuristic