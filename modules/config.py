import os
import json

class ConfigCollector:
	"""
	Description:
	    A class for collecting parameters of an experiment, creating a subfolder and storing the parameters there along
		with the result of the experiment

	Args:
	    expr_name: Name of the experiment
	    folder: Folder where to create the subfolder containing the results of the experiment
	    config_dict: dictionary of parameter of the experiment
	"""
	def __init__(self, folder, expr_name = 'experiment', config_dict = {}):
		self.config = config_dict
		self.folder = folder
		# find experiment id
		idx = []
		for f in os.listdir(self.folder):
			if f.startswith(expr_name):
				words = f.split('#')
				if len(words) > 0:
					try:
						idx.append(int(words[1]))
					except:
						pass
		if len(idx) > 0:
			id = max(idx) + 1
		else:
			id = 0
		self.expr_name = expr_name + '#' + str(id)
		# create subfolder to contain experiment results
		self.res_path = os.path.join(folder, self.expr_name)
		if not os.path.exists(self.res_path):
		    os.mkdir(self.res_path)

	def add_params(self, params):
		"""
		Description:
			Updates config with newly supplied dictionary
		Args:
			params: a dict containing new parameters
		"""
		self.config.update(params)

	def write(self, mode='json'):
		"""
		Description:
		    Creates a subfolder to store the results of an experiment and saves the configuration of the experiment in the folder
		"""
		# writes down the configuration of the experiment
		config_path = os.path.join(self.folder, self.expr_name, '')
		if mode == 'txt':
			with open(config_path + 'config.txt', 'w') as f:
				for k in sorted(self.config):
					f.write('{}: {}\n'.format(k, self.config[k]))
		elif mode == 'json':
			with open(config_path + 'config.json', 'w') as f:
				json.dump(self.config, f, indent=2)
