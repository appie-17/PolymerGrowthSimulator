import numpy as np
import pandas as pd
import simulation as sim
import matplotlib.pyplot as plt


# Create object defined by experimental data and simulation program to run
class Comparison:
    def __init__(self, file_name, simulation):
        
        self.sim = simulation
        self.exp_df = pd.read_excel(file_name)
        self.exp_molmass = self.exp_df[self.exp_df.columns[0]].values
        self.exp_values = self.exp_df[self.exp_df.columns[1]].values
        self.exp_chainlen = ((self.exp_molmass-180) / 99.13).astype(int)
        self.exp_cl_min = self.exp_chainlen.min()
        # self.exp_chainlen = np.concatenate((np.arange(1, self.exp_cl_min), self.exp_chainlen))
        self.exp_cl_val = dict(zip(self.exp_chainlen, self.exp_values))
        self.exp_cl_val.update(zip(np.arange(1, self.exp_cl_min + 1), np.zeros(self.exp_cl_min)))
        self.exp_val = np.array(list(self.exp_cl_val.values()))        
        self.exp_val = np.concatenate((np.zeros(self.exp_cl_min - 1), self.exp_val))        
        self.sigma = [1,3,5,5,5]
        self.fig, self.axes = plt.subplots(ncols=2)

    def get_difference(self):
    	pass

    # Input arguments to generate simulation results and return
    # difference with experimental data
    

class minMaxNorm(Comparison) :
	def __init__(self,file_name,simulation):
		super().__init__(file_name, simulation)
			
	def get_difference(self, arguments, plot=False):
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
			difference = np.sum(abs(exp_norm - sim_norm)) / (sim_norm_sum / exp_norm_sum) ** 2
		# difference = np.sum(np.sqrt((exp_norm - sim_norm)**2))/(sim_norm_sum/exp_norm_sum)**2		

		else:
			difference = np.sum(abs(exp_norm - sim_norm)) / (exp_norm_sum / sim_norm_sum) ** 2
		# difference = np.sum(np.sqrt((exp_norm - sim_norm)**2))/(exp_norm_sum/sim_norm_sum)**2		

		return difference
		# dead, living, coupled = [(180+99.13*x) for x in sim_data]	

class medianFoldNorm(Comparison) :
	def __init__(self, file_name, simulation):
		super().__init__(file_name, simulation)
	
	def get_difference(self, arguments, plot=False):
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
			sim_norm = sim_val*median_foldNorm
			
		elif diff <= 0:
			sim_val = np.concatenate((sim_val, np.zeros(abs(diff))))
			exp_val = self.exp_val					
			foldNorm = np.divide(sim_val,exp_val, out=np.zeros(exp_val.shape), where=exp_val!=0)
			median_foldNorm = np.median(foldNorm[foldNorm.nonzero()])								
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
		difference = 0
		for i in range(len(self.sigma)):
			indices = np.where((exp_norm>exp_mean-exp_sd*i)&(exp_norm<exp_mean+exp_sd*i))
			difference += np.sum(abs((exp_norm[indices] - sim_norm[indices]))**(1/self.sigma[i]))



		# Compute difference by l1- or l2-norm
		# if exp_norm_sum > sim_norm_sum:
		# 	difference = np.sum(abs((exp_norm - sim_norm)**2)) / (sim_norm_sum / exp_norm_sum)
		# # difference = np.sum(np.sqrt((exp_norm - sim_norm)**2))/(sim_norm_sum/exp_norm_sum)**2
		# #print(difference)

		# else:
		# 	difference = np.sum(abs((exp_norm - sim_norm)**2)) / (exp_norm_sum / sim_norm_sum)
		# difference = np.sum(np.sqrt((exp_norm - sim_norm)**2))/(exp_norm_sum/sim_norm_sum)**2
		#print(difference)

		return difference
		# dead, living, coupled = [(180+99.13*x) for x in sim_data]	