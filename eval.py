from simulation import polymer
from hillClimbing import hillClimbing
from distributionComparison import minMaxNorm, medianFoldNorm, distributionComparison
from evolutionaryAlgorithm import EvolutionaryAlgorithm
import numpy as np

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
    ar[6] = prob(non_neg(ar[6]))
    #  d_exponent --> non-negative
    ar[7] = prob(non_neg(ar[7]))
    #  l_naked --> lol
    ar[8] = prob(non_neg(ar[8]))
    #  kill_spawns_new --> boolean
    ar[9] = 1
    
    return ar

def process_arguments(arguments):
    arguments = clip(arguments)
    # print(arguments)
    return arguments

if __name__ == '__main__':
    #Compare distributions after normalizing by either minMaxNorm or medianFoldNorm. 
    #compareDist =  minMaxNorm('polymer_20k.xlsx', polymer)
    compareDist =  medianFoldNorm('polymer_30k.xlsx', polymer)
    
    #Use compare
    # hillClimbing(compareDist.costFunction, process_arguments)
    param_boundaries = np.array([[900,1100],[90000,110000],[3000000,32000000],
                   [0,1],[0,0.0001],[0,1],[0,1],[0,1],[0,1],[1,1]])
    alg = EvolutionaryAlgorithm(param_boundaries, 20, compareDist.costFunction, process_arguments, graph=True)
    alg.log_level = 2
    print(alg.run(100))
    print(alg.population)

    X0 = np.array([[1000, 100000, 31600000, 0.2,
    0.0000806, 0.5, 0.67, 0.67, 1, 1]])
    xp,yp = bayesian_optimisation(15,compareDist.costFunction, bound,
                        None,1,alpha=0.1,epsilon=1e-5)