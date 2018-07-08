import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



""" distributionComparison
	Parent class for any combination of normalization and cost function. 
	Also does some preprocessing to load an experimental dataset.
    

    Arguments:
    ----------
    	file_name: string.
    	Define location of .xls file containing experimental dataset
    	simulation: function.
    	Pointer towards simulation function returning
    	a simulated distribution of polymer lengths.
    	fig: boolean.
    	Plot a comparison between the experimental- and simulated distribution
    	of polymer lengths.
"""
class distributionComparison:
	def __init__(self, file_name, simulation, fig=None):

		self.sim = simulation 
		self.exp_df = pd.read_excel(file_name)
		self.exp_molmass = self.exp_df[self.exp_df.columns[0]].values #First column of the dataframe corresponding to the molar_mass
		self.exp_values = self.exp_df[self.exp_df.columns[1]].values #Values for  linear differential molar mass
		self.exp_chainlen = ((self.exp_molmass-180) / 99.13).astype(int) #Convert molar_mass to polymer lenghts
		self.cl_max = self.exp_chainlen[-1]
		self.exp_cl_val = dict(zip(
			np.arange(self.cl_max),
			np.zeros(self.cl_max))) #Dictionary with polymer lengths initialzed with zeros
		for cl in self.exp_cl_val: #Fill each polymer lengths with a linear differential molar mass
			ind = np.where(cl==self.exp_chainlen)
			if not len(ind[0])==0:
				self.exp_cl_val[cl] = self.exp_values[ind[0][0]]		
		
		for cl in self.exp_cl_val.keys():
			if (self.exp_cl_val[cl]==0) & (cl!=0):				
				
				self.exp_cl_val[cl] = self.exp_cl_val[cl-1]				

		self.exp_val = np.array(list(self.exp_cl_val.values()))
		



		if not fig: #Initialize plot figure
			self.fig = plt.figure(figsize=(15,5))
		else:
			self.fig = fig
		self.ax0, self.ax1, self.ax2 = self.fig.add_subplot(131), self.fig.add_subplot(132), self.fig.add_subplot(133)
		self.lowest_cost = np.inf #Set initial best value to infinity

	def preprocessDist(self, dead, living, coupled): #Preprocess distribution from simulation output
		sim_data = np.concatenate((dead, living, coupled))
		sim_cl_max = sim_data.max()
		sim_val, sim_bins = np.histogram(sim_data, bins=np.arange(sim_cl_max + 1))

		diff = int(sim_cl_max - self.exp_val.shape[0])

		if diff > 0:
			exp_val = np.concatenate((self.exp_val, np.zeros(abs(diff))))
		elif diff <= 0:
			sim_val = np.concatenate((sim_val, np.zeros(abs(diff))))
			exp_val = self.exp_val

		return exp_val, sim_val

	def costFunction(self, arguments, plot=False): 
		pass
	# Input arguments to generate simulation results and return
	# difference with experimental data

	def plotDistributions(self, exp_norm, sim_norm, cost): 
		self.ax0.clear()
		self.ax0.bar(np.arange(exp_norm.shape[0]), exp_norm)
		self.ax0.set_title('Experiment')
		self.ax1.clear()
		self.ax1.bar(np.arange(sim_norm.shape[0]), sim_norm)
		self.ax1.set_title('Simulation')
		if cost < self.lowest_cost:
			self.lowest_cost = cost
			self.ax2.clear()
			self.ax2.bar(np.arange(sim_norm.shape[0]), sim_norm)
			self.ax2.set_title('Best Match')
		plt.pause(1e-40)

""" minMaxNorm
	First implemented normalization and cost function

"""
class minMaxNorm(distributionComparison) :
	def __init__(self,file_name,simulation, fig=None):
		super().__init__(file_name, simulation, fig)
			
	def costFunction(self, arguments, plot=False):

		dead, living, coupled = self.sim(*arguments) #Run simulation polymer growth
		exp_val, sim_val = self.preprocessDist(dead, living, coupled)

		# Normalize both exp- and sim-data by min-max normalization
		exp_val_max = exp_val.max()
		exp_norm = exp_val / exp_val_max
		exp_norm_sum = np.sum(exp_norm)

		sim_val_max = sim_val.max()
		sim_norm = sim_val / sim_val_max
		sim_norm_sum = np.sum(sim_norm)


		# Compute difference by l2-norm
		if exp_norm_sum > sim_norm_sum:
			cost = np.sum(abs(exp_norm - sim_norm)) / (sim_norm_sum / exp_norm_sum) ** 2

		else:
			cost = np.sum(abs(exp_norm - sim_norm)) / (exp_norm_sum / sim_norm_sum) ** 2

		if plot:
			# print(arguments)
			self.plotDistributions(exp_norm, sim_norm, cost)

		return cost

""" medianFoldNorm
	Second implemented normalization and cost function

"""
class medianFoldNorm(distributionComparison) :
	def __init__(self, file_name, simulation, sigma=None, fig=None):
		super().__init__(file_name, simulation, fig)
		self.median_foldNorm=1 #Initial median_foldNorm
		if sigma is None:
			self.sigma = [1,5,5,5,5,5] #Initial weights for each part of distribution
		else:
			self.sigma = sigma

	def costFunction(self, arguments, plot=False):
		
		dead, living, coupled = self.sim(*arguments)
		exp_val, sim_val = self.preprocessDist(dead, living, coupled)

		foldNorm = np.divide(exp_val,sim_val, out=np.zeros(sim_val.shape), where=sim_val!=0) #Division by 0 returns 0
		median_foldNorm = np.median(foldNorm[foldNorm.nonzero()]) #Compute median from all folds, excluding zero's
		if not np.isfinite(median_foldNorm): median_foldNorm=self.median_foldNorm #If no valid fold use previous foldNorm
		self.median_foldNorm = median_foldNorm

		sim_norm = sim_val*median_foldNorm #Normalize simulation values by medianFoldNorm

		exp_norm = exp_val

		#Cost function based on weights by standard deviation
		exp_sd, exp_mean = np.std(exp_norm), np.mean(exp_norm)
		cost = 0
		#Weight each part of the distribution
		for i in range(len(self.sigma)):
			indices = np.where((exp_norm>exp_mean-exp_sd*i)&(exp_norm<exp_mean+exp_sd*i))
			cost += np.sum(abs((exp_norm[indices] - sim_norm[indices]))**(1/self.sigma[i]))

		if plot:
			# print(arguments)
			self.plotDistributions(exp_norm, sim_norm, cost)
		return cost

""" translationInvariant
	Using above version of medianFoldNormalization with a different cost function.

"""

class translationInvariant(distributionComparison):
	def __init__(self, file_name, simulation, sigma=None, transfac=1, fig=None):
		super().__init__(file_name, simulation, fig)
		self.median_foldNorm = 1 #Initial medianFoldNorm
		if sigma is None:
			self.sigma = [1,1,1,1,1,1] #Initial weights for each part of distribution
		else:
			self.sigma = sigma
		self.transfac = transfac

	def costFunction(self, arguments, plot=False):

		dead, living, coupled = self.sim(*arguments)
		exp_val, sim_val = self.preprocessDist(dead, living, coupled)

		posmaxsim = np.where(sim_val == sim_val.max())
		posmaxexp = np.where(exp_val == exp_val.max())
		percentage = abs((posmaxsim[0]/posmaxexp[0])-1) #measure relative distance of the peaks
		f = posmaxsim[0]-posmaxexp[0] #when negative move simulation data to the right. when positive move to the left
		if f[0]>= 0:#move simulation data to the left
			cutted_sim_val = sim_val[f[0]:]
			trans_sim_val = np.append(cutted_sim_val, np.zeros(f[0]))
		if f[0]<0: #move simulation data to the right
			cutted_sim_val = sim_val[:len(sim_val)+f[0]]
			trans_sim_val = np.append(np.zeros(abs(f[0])), cutted_sim_val)
		print(f[0])

		foldNorm = np.divide(exp_val, trans_sim_val, out=np.zeros(trans_sim_val.shape), where=trans_sim_val != 0) #Division by 0 returns 0
		median_foldNorm = np.median(foldNorm[foldNorm.nonzero()]) #Compute median from all folds, excluding zero's
		if not np.isfinite(median_foldNorm): median_foldNorm = self.median_foldNorm #If no valid fold use previous foldNorm
		self.median_foldNorm = median_foldNorm

		trans_sim_norm = trans_sim_val * median_foldNorm #Normalize trans_inv simulation values by medianFoldNorm
		# sim_norm = sim_val * median_foldNorm 

		exp_norm = exp_val

		# Cost function based on weights by standard deviation
		exp_sd, exp_mean = np.std(exp_norm), np.mean(exp_norm)
		cost = 0
		for i in range(len(self.sigma)):
			indices = np.where((exp_norm > exp_mean - exp_sd * i) & (exp_norm < exp_mean + exp_sd * i))
			cost += np.sum(abs((exp_norm[indices] - trans_sim_norm[indices])) ** (1 / self.sigma[i]))

		if plot:
			# print(arguments)
			self.plotDistributions(exp_norm, trans_sim_norm, cost)
		print(percentage)
		cost = cost * np.exp(percentage/self.transfac)
		print(cost)
		return cost