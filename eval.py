from simulation import polymer
from hill_climbing import hill_climbing
from data_processing import comparison
import numpy as np

diff =  comparison('polymer_20k.xlsx', polymer)

def non_neg(x):
    return np.sqrt(x**2)

def int_round(x):
    return np.round(x).astype(int)

def monom(x):
    if x < -1:
        return -1
    return x

def prob(x):
    if x < 0:
        return 0
    elif x > 1:
        return 0.99999
    return x

def clip(ar):
    ar = list(ar)
    # time_sim --> round + non-negative
    ar[0] = int_round(non_neg(ar[0]))
    #  number_of_molecules --> round + non-negative
    ar[1] = int_round(non_neg(ar[1]))
    #  monomer_pool --> round + -1?
    ar[2] = int_round(monom(ar[2]))
    #  p_growth -->  between 0 and 1 + non-negative
    ar[3] = prob(non_neg(ar[3]))
    #  p_death --> between 0 and 1 + non-negative
    ar[4] = prob(non_neg(ar[4]))
    #  p_dead_react --> between 0 and 1 + non-negative
    ar[5] = prob(non_neg(ar[5]))
    #  l_exponent --> non-negative
    ar[6] = non_neg(ar[6])
    #  d_exponent --> non-negative
    ar[7] = non_neg(ar[7])
    #  l_naked --> lol
    ar[8] = ar[8]
    #  kill_spawns_new --> boolean
    ar[9] = 0
    return ar

def process_arguments(arguments):
    arguments = clip(arguments)
    # print(arguments)
    return arguments

if __name__ == '__main__':
    hill_climbing(diff.get_difference, process_arguments)