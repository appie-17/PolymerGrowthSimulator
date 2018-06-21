import numpy as np
import pandas as pd
import simulation as sim
import matplotlib.pyplot as plt

"""
Comparison class is the parent class for any normalization method to be implemented.
Inherit the 'Comparison' class to load and preprocess experimental datasets from a 2 column excel file,
First column is the molar_mass (g/mol) and second column the linear differential molar mass.
Each normalization method is forced to implement the abstract method 'costFunction', which returns a final cost
when comparing two distributions.
"""

class distributionComparison:
	def __init__(self, file_name, simulation):

		self.sim = simulation
		self.exp_df = pd.read_excel(file_name)
		self.exp_molmass = self.exp_df[self.exp_df.columns[0]].values
		self.exp_values = self.exp_df[self.exp_df.columns[1]].values
		self.exp_chainlen = ((self.exp_molmass-180) / 99.13).astype(int)
		self.exp_cl_min = self.exp_chainlen.min()
		self.exp_cl_val = dict(zip(self.exp_chainlen, self.exp_values))
		self.exp_cl_val.update(zip(np.arange(1, self.exp_cl_min + 1), np.zeros(self.exp_cl_min)))
		self.exp_val = np.array(list(self.exp_cl_val.values()))
		self.exp_val = np.concatenate((np.zeros(self.exp_cl_min - 1), self.exp_val))
		self.sigma = [1,3,5,5,5]
		self.fig, self.axes = plt.subplots(ncols=2)


	def costFunction(self):
		pass

	# Input arguments to generate simulation results and return
	# difference with experimental data


class minMaxNorm(distributionComparison) :
	def __init__(self,file_name,simulation):
		super().__init__(file_name, simulation)
			
	def costFunction(self, arguments, plot=False):
		dead, living, coupled = self.sim(*arguments)
		sim_data = np.concatenate((dead, living, coupled))
		sim_cl_max = sim_data.max()	
		sim_val, sim_bins = np.histogram(sim_data, bins=np.arange(sim_cl_max + 1))
		diff = int(sim_cl_max - self.exp_val.shape[0])	

		if diff > 0:
			exp_val = np.concatenate((self.exp_val, np.zeros(abs(diff))))
		elif diff < 0:
			sim_val = np.concatenate((sim_val, np.zeros(abs(diff))))
			exp_val = self.exp_val

		# Normalize both exp- and sim-data by min-max normalization
		exp_val_max = exp_val.max()
		exp_norm = (exp_val) / exp_val_max
		exp_norm_sum = np.sum(exp_norm)

		sim_val_max = sim_val.max()
		sim_norm = (sim_val) / sim_val_max
		sim_norm_sum = np.sum(sim_norm)

		# Plot difference after certain number of parameter iterations
		if plot:				
			plt.figure(self.fig.number)
			self.axes[0].clear()
			self.axes[0].bar(np.arange(exp_norm.shape[0]), exp_norm)
			self.axes[0].set_title('Experiment')
			self.axes[1].clear()
			self.axes[1].bar(np.arange(sim_norm.shape[0]), sim_norm)
			self.axes[1].set_title('Simulation')
			plt.pause(1e-40)								

		# Compute difference by l1- or l2-norm
		if exp_norm_sum > sim_norm_sum:
			cost = np.sum(abs(exp_norm - sim_norm)) / (sim_norm_sum / exp_norm_sum) ** 2
		# cost = np.sum(np.sqrt((exp_norm - sim_norm)**2))/(sim_norm_sum/exp_norm_sum)**2		

		else:
			cost = np.sum(abs(exp_norm - sim_norm)) / (exp_norm_sum / sim_norm_sum) ** 2
		# cost = np.sum(np.sqrt((exp_norm - sim_norm)**2))/(exp_norm_sum/sim_norm_sum)**2		

		return cost
		# dead, living, coupled = [(180+99.13*x) for x in sim_data]	

class medianFoldNorm(distributionComparison) :
	def __init__(self, file_name, simulation):
		super().__init__(file_name, simulation)
	
	def costFunction(self, arguments, plot=True):
		dead, living, coupled = self.sim(*arguments)
		sim_data = np.concatenate((dead, living, coupled))
		sim_cl_max = sim_data.max()
		sim_val, sim_bins = np.histogram(sim_data, bins=np.arange(sim_cl_max + 1))

		diff = int(sim_cl_max - self.exp_val.shape[0])

		exp_val = self.exp_val

		# Normalize exp- and sim-data by median-fold normalization
		if diff > 0:
			exp_val = np.concatenate((self.exp_val, np.zeros(abs(diff))))
			foldNorm = np.divide(exp_val,sim_val, out=np.zeros(sim_val.shape), where=sim_val!=0)
			median_foldNorm = np.median(foldNorm[foldNorm.nonzero()])
			if not np.isfinite(median_foldNorm): median_foldNorm=1
			sim_norm = sim_val*median_foldNorm
			
		elif diff <= 0:
			sim_val = np.concatenate((sim_val, np.zeros(abs(diff))))
			exp_val = self.exp_val					
			foldNorm = np.divide(sim_val,exp_val, out=np.zeros(exp_val.shape), where=exp_val!=0)
			median_foldNorm = np.median(foldNorm[foldNorm.nonzero()])								
			if not np.isfinite(median_foldNorm): median_foldNorm=1
			sim_norm = sim_val/median_foldNorm
			
		sim_norm_sum = np.sum(sim_norm)
		exp_norm = exp_val
		exp_norm_sum = np.sum(exp_norm)
		
		# Plot difference after certain number of parameter iterations
		if plot:			
			plt.figure(self.fig.number)
			self.axes[0].clear()
			self.axes[0].bar(np.arange(exp_norm.shape[0]), exp_norm)
			self.axes[0].set_title('Experiment')

			self.axes[1].clear()
			self.axes[1].bar(np.arange(sim_norm.shape[0]), sim_norm)
			self.axes[1].set_title('Simulation')
			plt.pause(1e-40)

		#Cost function based on weights by standard deviation
		exp_sd, exp_mean = np.std(exp_norm), np.mean(exp_norm)
		cost = 0
		for i in range(len(self.sigma)):
			indices = np.where((exp_norm>exp_mean-exp_sd*i)&(exp_norm<exp_mean+exp_sd*i))
			cost += np.sum(abs((exp_norm[indices] - sim_norm[indices]))**(1/self.sigma[i]))


		# Compute cost by l1- or l2-norm
		# if exp_norm_sum > sim_norm_sum:
		# 	cost = np.sum(abs((exp_norm - sim_norm)**2)) / (sim_norm_sum / exp_norm_sum)
		# # cost = np.sum(np.sqrt((exp_norm - sim_norm)**2))/(sim_norm_sum/exp_norm_sum)**2
		# #print(cost)

		# else:
		# 	cost = np.sum(abs((exp_norm - sim_norm)**2)) / (exp_norm_sum / sim_norm_sum)
		# cost = np.sum(np.sqrt((exp_norm - sim_norm)**2))/(exp_norm_sum/sim_norm_sum)**2
		#print(cost)
		
		return cost
		# dead, living, coupled = [(180+99.13*x) for x in sim_data]


class Trans(distributionComparison):
	def __init__(self, file_name, simulation):
		super().__init__(file_name, simulation)

	def costFunction(self, arguments, plot=True):
		dead, living, coupled = self.sim(*arguments)
		sim_data = np.concatenate((dead, living, coupled))
		sim_cl_max = sim_data.max()
		sim_val, sim_bins = np.histogram(sim_data, bins=np.arange(sim_cl_max + 1))

		posmaxsim = np.where(sim_val.max())
		posmaxexp = np.where(self.exp_val.max())
		f = posmaxsim[0]-posmaxexp[0] #when negative move simulation data to the right. when positive move to the left
		if f> 0:#move simulation data to the left
			cutted_sim_val = sim_val[f:]
			sim_val =[cutted_sim_val, np.zeros(f)]
		if f<0: #move simulation data to the right
			cutted_sim_val = sim_val[:len[sim_val]+f]
			sim_val = [np.zeros(abs(f)), cutted_sim_val]

		diff = int(sim_cl_max - self.exp_val.shape[0])

		exp_val = self.exp_val

		# Normalize exp- and sim-data by median-fold normalization
		if diff > 0:
			exp_val = np.concatenate((self.exp_val, np.zeros(abs(diff))))
			foldNorm = np.divide(exp_val, sim_val, out=np.zeros(sim_val.shape), where=sim_val != 0)
			median_foldNorm = np.median(foldNorm[foldNorm.nonzero()])
			if not np.isfinite(median_foldNorm): median_foldNorm = 1
			sim_norm = sim_val * median_foldNorm

		elif diff <= 0:
			sim_val = np.concatenate((sim_val, np.zeros(abs(diff))))
			exp_val = self.exp_val
			foldNorm = np.divide(sim_val, exp_val, out=np.zeros(exp_val.shape), where=exp_val != 0)
			median_foldNorm = np.median(foldNorm[foldNorm.nonzero()])
			if not np.isfinite(median_foldNorm): median_foldNorm = 1
			sim_norm = sim_val / median_foldNorm

		sim_norm_sum = np.sum(sim_norm)
		exp_norm = exp_val
		exp_norm_sum = np.sum(exp_norm)

		# Plot difference after certain number of parameter iterations
		if plot:
			plt.figure(self.fig.number)
			self.axes[0].clear()
			self.axes[0].bar(np.arange(exp_norm.shape[0]), exp_norm)
			self.axes[0].set_title('Experiment')
			self.axes[1].clear()
			self.axes[1].bar(np.arange(sim_norm.shape[0]), sim_norm)
			self.axes[1].set_title('Simulation')
			plt.pause(1e-40)

		# Cost function based on weights by standard deviation
		exp_sd, exp_mean = np.std(exp_norm), np.mean(exp_norm)
		cost = 0
		for i in range(len(self.sigma)):
			indices = np.where((exp_norm > exp_mean - exp_sd * i) & (exp_norm < exp_mean + exp_sd * i))
			cost += np.sum(abs((exp_norm[indices] - sim_norm[indices])) ** (1 / self.sigma[i]))

		# Compute cost by l1- or l2-norm
		# if exp_norm_sum > sim_norm_sum:
		# 	cost = np.sum(abs((exp_norm - sim_norm)**2)) / (sim_norm_sum / exp_norm_sum)
		# # cost = np.sum(np.sqrt((exp_norm - sim_norm)**2))/(sim_norm_sum/exp_norm_sum)**2
		# #print(cost)

		# else:
		# 	cost = np.sum(abs((exp_norm - sim_norm)**2)) / (exp_norm_sum / sim_norm_sum)
		# cost = np.sum(np.sqrt((exp_norm - sim_norm)**2))/(exp_norm_sum/sim_norm_sum)**2
		# print(cost)


		return cost*(np.exp(abs(f/3)))
		# dead, living, coupled = [(180+99.13*x) for x in sim_data]