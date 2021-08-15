# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

# import remaining modules
import simulate as sm
import filter as fl
import numpy as np
import scipy
import plot as plot

"""
A 3D non-linear problem (L63)
"""

# creates a Model object to feed the filter / combine the models
def get_model(x0, size, prior_cov=1.0, obs_cov=0.1, shift=0.0, obs_gap=0.1):
    # set parameters
    sigma, rho, beta = 10.0, 28.0, 8.0/3,
    eps = 0.0
    zero2, id2, zero3, id3 =  np.zeros(2), np.identity(2), np.zeros(3), np.identity(3)
    shift = shift * np.ones(3)

    # assign an attractor point as the starting point
    def lorenz63_f(t, state):
        x, y, z = state
        return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])

    def lorenz_63(x):
        return scipy.integrate.solve_ivp(lorenz63_f, [0.0, obs_gap], x, method='RK45', t_eval=[obs_gap]).y.T[0]

    # create a deterministic Markov chain
    prior = sm.Simulation(algorithm = lambda *args: shift + np.random.multivariate_normal(x0, prior_cov*id3))
    process_noise = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(zero3, eps*id3))
    func_h = lambda k, x, noise: lorenz_63(x) + noise
    conditional_pdf_h = lambda k, x, past: scipy.stats.multivariate_normal.pdf(x, mean = func_h(k, past, zero3), cov = eps*id3)

    # define the observation model
    func_o = lambda k, x, noise: np.array([x[0], x[2]]) + noise
    observation_noise = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(zero2, obs_cov*id2))
    conditional_pdf_o = lambda k, y, condition: scipy.stats.multivariate_normal.pdf(y, mean = func_o(0, condition, zero2), cov = obs_cov*id2)

    # generates a trajectory according to the dynamic model
    def gen_path(x, length):
        path = np.zeros((length, 3), dtype = 'float64')
        path[0] = x
        for i in range(1, length):
            path[i] = func_h(0, path[i-1], zero3)
        return path

    mc = sm.DynamicModel(size=size, prior=prior, func=func_h, sigma=eps*id3, noise_sim=process_noise, conditional_pdf=conditional_pdf_h)
    om = sm.MeasurementModel(size=size, func=func_o, sigma=obs_cov*id2, noise_sim=observation_noise, conditional_pdf=conditional_pdf_o)
    return fl.Model(dynamic_model = mc, measurement_model = om), gen_path
