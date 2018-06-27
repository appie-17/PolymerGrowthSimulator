import numpy as np
from simulation import polymer
from distributionComparison import minMaxNorm, medianFoldNorm, translationInvariant
from bayesianOptimization import bayesianOptimisation
import pickle
from sklearn import gaussian_process


if __name__ == '__main__':
    # Compare distributions after normalizing by either minMaxNorm or medianFoldNorm.
    compareDist = medianFoldNorm('Data/polymer_50k.xlsx', polymer, [1,5,5,5,5,5])

    param_boundaries = np.array([[5000, 8000], [100000, 150000], [50000000, 100000000],
                                 [0, 1], [0, 0.0001], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1]])

    gp = gaussian_process.GaussianProcessRegressor(kernel=gaussian_process.kernels.Matern(length_scale=1),
                                            alpha=0.1,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

    xp, yp, ybest, model = bayesianOptimisation(n_iters=150, costFunction=compareDist.costFunction, bounds=param_boundaries,
                                         n_params=10,
                                         x0=None, n_pre_samples=10, gaussian_process=gp, alpha=0.1,
                                         acquisitionFunction='probability_improvement', epsilon=1e-5)
    # np.savetxt(fname='testitest', X=yp)
    np.save(file='eval1_50k_MAT_PI_ls_1', arr=ybest)
    pickle.dump(model, open("model1_50k_MAT_PI_ls_1.p", "wb"))

    xp, yp, ybest, model = bayesianOptimisation(n_iters=150, costFunction=compareDist.costFunction,
                                                bounds=param_boundaries,
                                                n_params=10,
                                                x0=None, n_pre_samples=10, gaussian_process=gp, alpha=0.1,
                                                acquisitionFunction='probability_improvement', epsilon=1e-5)
    # np.savetxt(fname='testitest', X=yp)
    np.save(file='eval2_50k_MAT_PI_ls_1', arr=ybest)
    pickle.dump(model, open("model2_50k_MAT_PI_ls_1.p", "wb"))

    xp, yp, ybest, model = bayesianOptimisation(n_iters=150, costFunction=compareDist.costFunction,
                                                bounds=param_boundaries,
                                                n_params=10,
                                                x0=None, n_pre_samples=10, gaussian_process=gp, alpha=0.1,
                                                acquisitionFunction='expected_improvement', epsilon=1e-5)
    # np.savetxt(fname='testitest', X=yp)
    np.save(file='eval3_50k_MAT_EI_ls_1', arr=ybest)
    pickle.dump(model, open("model3_50k_MAT_EI_ls_1.p", "wb"))

    xp, yp, ybest, model = bayesianOptimisation(n_iters=150, costFunction=compareDist.costFunction,
                                                bounds=param_boundaries,
                                                n_params=10,
                                                x0=None, n_pre_samples=10, gaussian_process=gp, alpha=0.1,
                                                acquisitionFunction='expected_improvement', epsilon=1e-5)
    # np.savetxt(fname='testitest', X=yp)
    np.save(file='eval4_50k_MAT_EI_ls_1', arr=ybest)
    pickle.dump(model, open("model4_50k_MAT_EI_ls_1.p", "wb"))


