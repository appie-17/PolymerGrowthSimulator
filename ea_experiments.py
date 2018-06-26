from distributionComparison import medianFoldNorm, translationInvariant
from simulation import polymer
import numpy as np
from evolutionaryAlgorithm import EvolutionaryAlgorithm


file_name = "Data\polymer_20k.xlsx"
TI = translationInvariant(file_name, polymer, [1,3,5,5,5,5])
MFC = medianFoldNorm(file_name, polymer)


lower = [1000, 100000, 25000000, 0, 0,      0, 0, 0, 0, 1]
upper = [2000, 200000, 35000000, 1, 0.0005, 1, 1, 1, 1, 1]
bounds = np.stack((lower, upper), 1)


runs = 4
def pop_size_experiment():
    iterations = 20
    fitnesses = []
    sizes = [5, 10]#, 30]
    for pop in sizes:
        print("size", pop)
        run_results = []
        for run in range(runs):
            ea = EvolutionaryAlgorithm(bounds, pop, MFC.costFunction, ui_plot=False)
            fit = ea.run(iterations)
            run_results.append(np.mean(fit, 1))
        stacked = np.stack(run_results)
        fitnesses.append(np.mean(stacked, 0))

    return sizes, fitnesses

def mutation_experiment():
    iterations = 20
    fitnesses = []
    mutation_rates = [0.2, 0.4, 0.6]
    for mutation_rate in mutation_rates:
        print("mutation rate", mutation_rate)
        run_results = []
        for run in range(runs):
            ea = EvolutionaryAlgorithm(bounds, 10, MFC.costFunction, ui_plot=False, mutation_rate=mutation_rate)
            fit = ea.run(iterations)
            run_results.append(np.mean(fit, 1))
        stacked = np.stack(run_results)
        fitnesses.append(np.mean(stacked, 0))

    return sizes, fitnesses

def N_experiment():
    iterations = 20
    fitnesses = []
    N = [2, 4, 8]
    for n in N:
        print("n", n)
        run_results = []
        for run in range(runs):
            ea = EvolutionaryAlgorithm(bounds, 10, MFC.costFunction, ui_plot=False, n=n)
            fit = ea.run(iterations)
            run_results.append(np.mean(fit, 1))
        stacked = np.stack(run_results)
        fitnesses.append(np.mean(stacked, 0))

    return sizes, fitnesses

def iter_experiment():
    fitnesses = []
    iteration = 100
    run_results = []
    for run in range(runs):
        ea = EvolutionaryAlgorithm(bounds, 20, MFC.costFunction, ui_plot=False)
        fit = ea.run(iteration)
        #average over iterations
        run_results.append(np.mean(fit, 1))
    # average over runs
    stacked = np.stack(run_results)
    fitnesses.append(np.mean(stacked, 0))

    return iteration, fitnesses


if __name__ == '__main__':
    # iterations, res = iter_experiment()
    #
    # np.savetxt("iteration_experiment.csv", res, delimiter=",")

    sizes, results = pop_size_experiment()
    np.savetxt("size_experiment.csv", results, delimiter=",")

    # n_s, n_result = N_experiment()
    # np.savetxt("N_experiment.csv", n_result, delimiter=",")

    # rates, mutation_result = mutation_experiment()
    # np.savetxt("mutation_experiment", mutation_result, delimiter=",")