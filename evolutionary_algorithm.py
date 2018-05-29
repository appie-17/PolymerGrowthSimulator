import numpy as np
from matplotlib import pyplot as plt
from tkinter import TclError
from multiprocessing import Pool

p = Pool(4)


def _async_evaluate(population, fit_func):
    # Evaluate individual i
    # print(("Evaluating individual {}: {}".format(i, population[i])))
    x = p.map(fit_func, population)
    # fitness = fit_func(population[i])
    print("Fitness:{}".format(x))
    return x


class EvolutionaryAlgorithm:

    def __init__(self, pop_size, fitness_function, clipping_function=None, graph=False):
        # time_sim, number_of_molecules, monomer_pool, p_growth, p_death, p_dead_react,
        # l_exponent, d_exponent, l_naked, kill_spawns_new
        init = np.array([1000,    100000,     10000000,   0.8,
                            0.000005,    0.2,    0.23,    0.23,    0.5,    1])
        # init = np.array([10000, 100000, 10000000, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1])
        # init = np.array([1000, 100000,     31600000,   0.2,
    # 0.0000806, 0.5, 0.67, 0.67, 1, 1])
        # self.scale = np.array([0,     0,      1000,    0.05,
        #                     0.005,   0.05,   0.02,   0.02,   0.02,   0.1])
        self.scale = init*0.5

        # scale = np.array([1000,     10000,      1000000,    0.05,
        #                     0.05,   0.05,   0.02,   0.02,   0.02,   0.1])
        scale = init * .05

        # Init and scale are used to make the initial population.
        # init is used as the location for the normal distribution
        # and scale determines the width of the gaussian to sample

        # init = np.random.normal(scale=10, size=pop_size)
        # scale = np.abs(init * 0.2)

        # self.population = (np.random.random((pop_size, 10)) - 0.5) * scale + init # Uniformly distributed init
        self.population = np.random.normal(loc=init, scale=scale, size=(10, 10))  # normally distributed init

        # Member definitions
        self.fit_func = fitness_function
        self.pop_size = pop_size
        self.scale = scale
        self.clip_func = clipping_function
        self.graph = graph
        # Indicator for how verbose the EA should be 3 is print everything --> debug. and 0 is no output
        self.log_level = 0

        # Use a argument clipping function if available
        if self.clip_func is not None:
            self.clip()

    def set_population(self, pop):
        self.population = pop
        self.pop_size = len(pop)

    # Do not use
    def _log(self, message, level):
        if level <= self.log_level:
            print(message)

    # for printing important messages
    def log(self, message):
        self._log(message, 1)

    # For printing informational messages
    def info(self, message):
        self._log(message, 2)

    # For printing trace info. i.e telling you precisely what the algorithm is doing.
    def trace(self, message):
        self._log(message, 3)

    # Runs the EA for i iterations
    def run(self, iterations: int):
        # Array for storing the fitnesses for all iterations
        fitnessess = np.zeros((iterations, self.pop_size))
        for i in range(iterations):
            
            self.log("#### iteration {} ####".format(i))

            # Evaluate individuals and store fitness values
            fitness = self.evaluation()
            fitnessess[i] = fitness
            self.selection(fitness)
            self.reproduction()
            self.mutation()

            # Show the average and max fitness values per iterationif self.graph:
            plt.figure(2)
            plt.subplot(111)
            plt.cla()
            min_fit = np.min(fitnessess, 1)
            averages = np.average(fitnessess, 1)
            plt.plot(averages[:i+1])
            plt.plot(min_fit)
            # plt.subplot(111)
            try:
                plt.pause(1e-40)
            except TclError:
                pass

        return fitnessess

    # Evaluation of the population
    def evaluation(self):

        size = len(self.population)

        if True:
            fitness = np.array(_async_evaluate(self.population, self.fit_func))
        else:
            for i in range(size):
                fitness = np.zeros(size)
                self.trace("Evaluating individual {}: {}".format(i, self.population[i]))
                fitness[i] = self.fit_func(self.population[i])
                self.trace("Fitness:{}".format(fitness[i]))
        self.info("Average fitness: {}".format(np.average(fitness)))
        return fitness



    # Truncated ranked selection
    # TODO: Encapsulate this!
    def selection(self, fitness):
        n = 8
        order = np.argsort(fitness)
        order = order[:n]
        print(self.population[order[-1]])
        self.fit_func(self.population[order[-1]], plot=True)
        self.population = self.population[order]

    # Copy remaining population until population is at pop_size
    def reproduction(self):
        size = len(self.population)
        repeats = np.ceil(self.pop_size/size)
        pop = np.repeat(self.population, repeats, axis=0)
        self.population = pop[:self.pop_size]

    # clips the population into the allowable ranges
    def clip(self):
        self.population = np.apply_along_axis(func1d=self.clip_func, axis=1, arr=self.population)

    def mutation(self):
        # crossover
        crossover_rate = 0.3
        if np.random.random() < crossover_rate:
            # Pick 2 random individuals
            parents = np.random.randint(0, self.pop_size, size=2)
            # Pick random column
            crossover_point = np.random.randint(1, 10)
            # create 2 new individuals and replace parents in the population
            ind1 = np.append(self.population[parents[0], :crossover_point],
                             self.population[parents[1], crossover_point:])

            ind2 = np.append(self.population[parents[1], :crossover_point],
                             self.population[parents[0], crossover_point:])

            self.population[parents[0]] = ind1
            self.population[parents[1]] = ind2

        # mutation
        # TODO Change this make the chance of evolving 0.2 for all instead of for each
        mutation_rate = 0.2
        x = np.random.random(self.pop_size)
        mask = np.argwhere(x < mutation_rate)
        cols = np.random.choice(10,len(mask),replace=True)
        mutations = np.random.normal(scale=self.scale[cols])
      
        self.population[mask, cols] = self.population[mask, cols] + mutations
        if self.clip_func is not None:
            self.clip()




if __name__ == '__main__':
    from eval import process_arguments
    from simulation import polymer
    from data_processing import minMaxNorm
    from simulation import polymer

    diff = minMaxNorm("polymer_20k.xlsx", polymer )
    alg = EvolutionaryAlgorithm(10, diff.get_difference, process_arguments)
    alg.log_level = 3
    print(alg.run(10))
    for ind in alg.population:
        dat = polymer(*process_arguments(ind))
        plt.plot()
    plt.pause(-1)
    print(alg.population)