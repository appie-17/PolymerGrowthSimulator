import numpy as np
from matplotlib import pyplot as plt


class EvolutionaryAlgorithm:

    def __init__(self, pop_size, fitness_function, clipping_function=None, graph=False):
        # time_sim, number_of_molecules, monomer_pool, p_growth, p_death, p_dead_react,
        # l_exponent, d_exponent, l_naked, kill_spawns_new
        init = np.array([[1000,    100000,     10000000,   0.5,
                            0.5,    0.5,    0.2,    0.2,    0.2,    1]])

        scale = np.array([100,     10000,      1000000,    0.05,
                            0.05,   0.05,   0.02,   0.02,   0.02,   0.1])

        # init = np.random.normal(scale=10, size=pop_size)
        # scale = np.abs(init * 0.2)

        self.population = (np.random.random((pop_size, 10)) - 0.5) * scale + init
        self.fit_func = fitness_function
        self.pop_size = pop_size
        self.scale = scale
        self.clip_func = clipping_function
        self.graph = graph
        self.log_level = 0

        if self.clip_func is not None:
            self.clip()

    def _log(self, message, level):
        if level <= self.log_level:
            print(message)

    def log(self, message):
        self._log(message, 1)

    def info(self, message):
        self._log(message, 2)

    def trace(self, message):
        self._log(message, 3)

    def run(self, iterations: int):

        fitnessess = np.zeros((iterations, self.pop_size))
        for i in range(iterations):
            self.log("#### iteration {} ####".format(i))
            fitness = self.evaluation()
            fitnessess[i] = fitness
            self.selection(fitness)
            self.reproduction()
            self.mutation()

        if self.graph:
            plt.ion()
            averages = np.average(fitnessess, 1)
            plot = plt.plot(averages)
            plt.pause(-1)

        return fitnessess

    def evaluation(self):
        size = len(self.population)
        fitness = np.zeros(size)
        for i in range(size):
            self.trace("Evaluating individual {}: {}".format(i, self.population[i]))
            fitness[i] = self.fit_func(self.population[i])
            self.trace("Fitness:{}".format(fitness[i]))
        self.info("Average fitness: {}".format(np.average(fitness)))
        return fitness

    def selection(self, fitness):
        n = 3
        order = np.argsort(fitness)
        order = order[:n]
        self.population = self.population[order]

    def reproduction(self):
        size = len(self.population)
        repeats = np.ceil(self.pop_size/size)
        pop = np.repeat(self.population, repeats, axis=0)
        self.population = pop[:self.pop_size]

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
        mutation_rate = 0.2
        x = np.random.random(self.pop_size)
        mask = np.argwhere(x < mutation_rate)
        cols = np.random.randint(0, 10, len(mask))
        mutations = np.random.normal(scale=self.scale[cols])
        self.population[mask, cols] = self.population[mask, cols] + mutations
        if self.clip_func is not None:
            self.clip()


def rosenbrock(X):
    _a = 0
    _b = 10
    s = 0
    for i in range(len(X) - 1):
        s = s + _b * pow(X[i+1] - pow(X[i], 2), 2) + pow(_a-X[i], 2)
    return s


def test_fitness(X):
    return np.random.random()


if __name__ == '__main__':
    from eval import process_arguments
    from simulation import polymer
    from data_processing import comparison
    from simulation import polymer

    diff = comparison("polymer_20k.xlsx", polymer, )

    alg = EvolutionaryAlgorithm(10, diff.get_difference, process_arguments, True)
    alg.log_level = 2
    print(alg.run(200))
    print(alg.population)