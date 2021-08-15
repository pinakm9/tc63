# A helper module for various sub-tasks
from time import time
import numpy as np
import random
import contextlib
import os, io
import sys

def timer(func):
	"""
	Timing wrapper for a generic function.
	Prints the time spent inside the function to the output.
	"""
	def new_func(*args, **kwargs):
		start = time()
		val = func(*args,**kwargs)
		end = time()
		print('Time taken by {} is {} seconds'.format(func.__name__, end-start))
		return val
	return new_func

def delta(x, x_0):
	"""
	Description:
		Dirac delta function

	Args:
		x: input
		x_0: point where the mass is located

	Returns:
	 	eiter 0.0 or 1.0
	"""
	return 1.0 if np.array_equal(x, x_0) else 0.0

class Picker(object):
    """
    A class defining an object-picker from an array
    """
    def __init__(self, array):
        """
        array = array of objects to pick from
        """
        self.array = array

    def equidistant(self, objs_to_pick, start_pt = 0):
        """
		Description:
        	Picks objs_to_pick equidistant objects starting at the location start_pt
        Returns:
			the picked objects
        """
        increment = int((len(self.array) - start_pt)/objs_to_pick)
        if increment < 1:
            return self.array
        else:
            new_array = [0]*objs_to_pick
            j = start_pt
            for i in range(objs_to_pick):
                new_array[i] = self.array[j]
                j += increment
        return np.array(new_array)

def normalize_small(numbers, threshold = 50):
	log_numbers = [np.log(number) for number in numbers]
	max_log = np.max(log_numbers)
	for i, number in enumerate(numbers):
		if max_log - log_numbers[i] > threshold:
			number[i] = 0.0

def KL_div_MC(p, q, samples):
	result = 0.0
	for x in samples:
		px = p(x)
		result += px*np.log(px/q(x))
	return result/len(samples)

def TV_dist_MC(p, q, samples):
	result = 0.0
	for x in samples:
		result +=np.abs(p(x)-q(x))
	return 0.5*result/len(samples)

def TV_dist_MC_avg(p, q, samples, batch):
	dist = 0.0
	for i in range(int(len(samples)/batch)):
		result = 0.0
		for x in samples[i*batch: (i+1)*batch]:
			result +=np.abs(p(x)-q(x))
		dist += result/batch
	return 0.5*dist/(len(samples)/batch)


@contextlib.contextmanager
def silencer():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout