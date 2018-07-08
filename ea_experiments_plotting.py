import numpy as np
import matplotlib.pyplot as plt


def plot_results(res):
    file = res["file"]
    data = np.loadtxt(file, skiprows=1, delimiter=",")
    plt.plot(data[:, 1:], alpha=0.8, linestyle="-")
    try:
        s = res["scale"]
        plt.yscale(s)
    except KeyError:
        plt.yscale("linear")
    plt.title(res["title"])
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(res["labels"])
    plt.show(block=True)
    plt.pause(0.0001)


if __name__ == '__main__':
    res = [
        {"file": "output/Iterations.csv", "title": "Average fitness per generation",
         "labels": ["run" + str(x) for x in range(4)]},
        {"file": "output/Mutation.csv", "title": "Average fitness for mutation rate", "labels":["0.2", "0.4", "0.6"], "scale":"log"},
        {"file": "output/N.csv", "title": "Average fitness for N", "labels":["2","4", "8"], "scale":"log"},
        {"file": "output/Popsize.csv", "title": "Average fitness for population size", "labels":["5", "10", "30"], "scale":"log"}
    ]
    for r in res:
        plot_results(r)
    plt.show()
