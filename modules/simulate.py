import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.stats
from statsmodels.distributions.empirical_distribution import ECDF
import utility as ut

#####################################
# Random variable class definitions #
#####################################

class RVContinuous(object):
    """
    Description:
        This is a class for defining generic continuous random variables.

    Attributes:
        name: name of the random variable, default = 'unknown'
        a: left endpoint of support interval of pdf, default = 0.0
        b: right endpoint of support interval of pdf, default = 1.0
        params: parameters of cdf and pdf
        pdf_: family of pdfs without parmeters specified (exists if __init__ is given a pdf that is not None)
        cdf_: family of cdfs without parmeters specified (exists if __init__ is given a cdf that is not None)
        pdf: pdf with parameters specified (exists if __init__ is given a pdf that is not None)
        cdf: cdf with parameters specified (exists if __init__ is given a cdf that is not None)
        find_mean_: family of user-defined mean-finders without parameters specified (exists if __init__ is given a find_mean that is not None)
        find_var_: family of user-defined variance-finders without parameters specified (exists if __init__ is given a find_var that is not None)
        find_mean: family of user-defined mean-finder with parameters specified (exists if __init__ is given a find_mean that is not None)
        find_var: family of user-defined variance-finder with parameters specified (exists if __init__ is given a find_var that is not None)
        mean: mean of the distribution, default = 'not_yet_computed'
        var: variance of the distribution, default = 'not_yet_computed'

    Methods:
        set_params: resets parameters of the distribution
        compute_mean: computes and sets self.mean
        compute_var: computes and sets self.var
        set_stats: computes and sets the user-chosen statistics of the distribution using the easiest possible methods
                   depending on availability of find_mean, find_var etc
        set_unset_stats: sets only unset statistics of the distribution using self.set_stats
    """

    def __init__(self, name = 'unknown', support = (-np.inf, np.inf), cdf = None, pdf = None, find_mean = None, find_var = None, **params):
        """
        Args:
            name: name of the random variable
            support: support of the pdf, default = (-np.inf, np.inf)
            find_mean: custom function for computing mean, accpets parameters of the distribution as **kwargs
            find_var: custom function for computing variance, accpets parameters of the distribution as **kwargs
            params: dict of keyword arguments that are passed to pdf, cdf and inv_cdf

        Notes:
            Either pdf or cdf is required for mean and variance computation. One of them can be omitted.
            In case the random variable has a well-known distribution, providing the name of the random variable and
            **params = parameters of the distribution will set all other arguments automatically.
            Currently a known name can be anything in the list ['gamma']. Dafault is 'unknown'.
        """
        # set support and pdf/cdf for known distributions
        if name == 'gamma':
            support = (0.0, np.inf)
            cdf = lambda x, shape, scale: scipy.stats.gamma.cdf(x, shape = shape, scale = scale)
            pdf = lambda x, shape, scale: scipy.stats.gamma.pdf(x, shape = shape, scale = scale)
        elif name == 'normal':
            cdf = lambda x, mean, cov: scipy.stats.multivariate_normal.cdf(x, mean = mean, cov = cov)
            pdf = lambda x, mean, cov: scipy.stats.multivariate_normal.pdf(x, mean = mean, cov = cov)

        # assign basic attributes
        self.name = name # name of the random variable
        self.a, self.b = support # left and right endpoints of support interval of pdf
        self.params = params # parameters of pdf and cdf
        if pdf is not None:
            self.pdf_ = pdf # family of pdfs without parmeters specified
            self.pdf = lambda x: self.pdf_(x, **self.params) # pdf with parameters specified
        if cdf is not None:
            self.cdf_ = cdf # family of cdfs without parmeters specified
            self.cdf = lambda x: self.cdf_(x, **self.params) # cdf with parameters specified
        if find_mean is not None:
            self.find_mean_ = find_mean # family of find_means without parmeters specified
            self.find_mean = lambda: self.find_mean_(**self.params) # find_mean with parameters specified
        if find_var is not None:
            self.find_var_ = find_var # family of find_vars without parmeters specified
            self.find_var = lambda: self.find_var_(**self.params) # find_var with parameters specified)
        self.mean = 'not_yet_computed'
        self.var = 'not_yet_computed'


    def set_params(self, **new_params):
        """
        Description:
            Resets parameters of the distribution to new_params.
            Passing only the parameters that need to be changed suffices.
        """
        for key, value in new_params.items():
            self.params[key] = value
        if hasattr(self, 'pdf'):
            self.pdf = lambda x: self.pdf_(x, **self.params)
        if hasattr(self, 'cdf'):
            self.cdf = lambda x: self.cdf_(x, **self.params)
        if hasattr(self, 'find_mean'):
            self.find_mean = lambda: self.find_mean_(**self.params)
        if hasattr(self, 'find_var'):
            self.find_var = lambda: self.find_var_(**self.params)


    def compute_mean(self):
        """
        Description:
            Computes and sets self.mean = expected value of the random variable.
        """
        # compute mean according to availability of pdf or cdf
        if hasattr(self, 'pdf'):
            # compute mean using pdf
            self.mean = integrate.quad(lambda x: x*self.pdf(x), self.a, self.b)[0]
        elif hasattr(self, 'cdf'):
            # parts of integrand
            plus = lambda x: 1.0 - self.cdf(x)
            minus = lambda x: -self.cdf(x)

            # left limit of integration
            left_lim = self.a

            # decide integrand and correction term according to self.a and self.b being finite/infinite
            if not np.isinf(self.a):
                integrand = plus
                correction = self.a
            elif not np.isinf(self.b):
                integrand = minus
                correction = self.b
            else:
                integrand = lambda x: plus(x) + minus(-x)
                correction = 0.0
                left_lim = 0.0

            # compute mean using cdf
            self.mean = integrate.quad(integrand, left_lim, self.b)[0] + correction
        else:
            # if no pdf or cdf is defined, return nan
            self.mean = float('NaN')
        return self.mean

    def compute_var(self):
        """
        Description:
            Computes and sets self.var = variance of the random variable.
        """
        # compute variance according to availability of pdf or cdf
        if hasattr(self, 'pdf'):
            # compute variance using pdf
            self.var = integrate.quad(lambda x: x*x*self.pdf(x), self.a, self.b)[0] - self.mean**2
        elif hasattr(self, 'cdf'):
            # parts of integrand
            plus = lambda x: 2.0*x*(1.0 - self.cdf(x))
            minus = lambda x: 2.0*x*self.cdf(x)

            # left limit of integration
            left_lim = self.a

            # decide integrand and correction term according to self.a and self.b being finite/infinite
            if not np.isinf(self.a):
                integrand = plus
                correction = self.a**2 - self.mean**2
            elif not np.isinf(self.b):
                integrand = minus
                correction = self.b**2 - self.mean**2
            else:
                integrand = lambda x: plus(x) - minus(-x)
                correction = - self.mean**2
                left_lim = 0.0

            # compute variance using cdf
            self.var = integrate.quad(integrand, left_lim, self.b)[0] + correction
        else:
            # if no pdf or cdf is defined, return nan
            self.var = float('NaN')
        return self.var

    def set_stats(self, stats = ()):
        """
        Description:
            Computes and sets the user-chosen statistics of the distribution using the easiest possible methods
            depending on availability of find_mean, find_var etc.

        Args:
            stats: list/tuple of statistic names to be computed.

        Notes:
            If the value is set to True, set_stats will try to compute the corresponding statistic.
            If stats = () (default), all statistics are computed.
        """
        for stat in stats:
            if hasattr(self, 'find_' + stat):
                setattr(self, stat, getattr(self, 'find_' + stat)())
            else:
                setattr(self, stat, getattr(self, 'compute_' + stat)())

    def set_unset_stats(self, stats = ()):
        """
        Description:
        Sets only unset statistics of the distribution using self.set_stats.

        Args:
            stats: list/tuple of unset statistics.
            In case stats = () (default), all unset statistics are set.
        """
        if stats == ():
            stats = ('mean', 'var')
        stats_to_compute = []
        for stat in stats:
            if hasattr(self, stat):
                stats_to_compute.append(stat)
        self.set_stats(stats_to_compute)


#############################################
# Sampling algorithm (function) definitions #
#############################################

def inverse_transform(inv_cdf, **params):
    """
    Description:
        The inverse transform algorithm for sampling.

    Arguments:
        inv_cdf: inverse of the cdf of the random variable to be sampled
        params: dict of parameters of the distribution

    Returns:
        the generated sample
    """
    return inv_cdf(np.random.uniform(), **params)

def composition(sim_components, probabilties):
    """
    Description:
        The composition technique for sampling.

    Args:
        sim_components: list of simulations
        probabilties: a discrete probability distribution

    Returns:
        the generated sample
    """
    return sim_components[np.random.choice(len(sim_components), p = probabilties)].algorithm()

def rejection(target_rv, helper_sim, ratio_bound):
    """
    Description:
        The accept-reject method for sampling.

    Args:
        target_rv: target random variable.
        helper_sim: simulation for helper random variable with pdf assigned.
        ratio_bound: an upper bound for the ratio of the pdfs.

    Returns:
        the generated sample
    """
    while True:
        sample = helper_sim.algorithm()
        if np.random.uniform() <= target_rv.pdf(sample)/(ratio_bound*helper_sim.rv.pdf(sample)):
            return sample


################################
# Simulation class definitions #
################################

class Simulation(object):
    """
    Description:
        This is a class for simulating a random variable X: Omega -> R^d.

    Attributes:
        rv: random variable to simulate, default = None
        #current_value: last simulated value of self.rv
        algorithm: function that returns a single smaple
        dimension: dimension of the codomain of the random variable d
        algorithm_args: keyword arguments for constructing self.algorithm which produces a single sample
        ecdf: empirical distribution of the generated samples (created during the first call to compare)

    Methods:
        generate: generates a batch of samples using self.algorithm
        compare: draws cdfs for target and simulation and sets self.ecdf
        set_algorithm: constructs self.algorithm

    Notes:
        self.algorithm only accepts *args as its arguments
    """

    def __init__(self, target_rv = None, algorithm = lambda *x: 0.0, *args, **algorithm_args):
        """
        Args:
            target_rv: random variable to simulate
            algorithm: a function that produces a single sample of target_rv
            algorithm_args: dict of keyword arguments that are passed to algorithm
        """
        # assign basic attributes
        self.rv = target_rv # random variable to simulate
        self.current_value = None # last simulated value of target_rv
        self.algorithm_args = algorithm_args # keyword arguments for constructing self.algorithm which produces a single sample
        # self.uniform  = np.random.uniform # uniform distribution, needed for multiprocessing compatibility
        # self.choice = np.random.choice # discrete distribution, needed for multiprocessing compatibility
        # self.gamma = np.random.gamma # gamma distribution, needed for multiprocessing compatibility
        self.set_algorithm(algorithm, *args, **algorithm_args)

    def generate(self, sample_size = 1, *args):
        """
        Description:
            Generates a batch of samples using self.algorithm
            args are the arguments that are passed to self.algorithm ???

        Returns:
            self.samples
        """
        self.size = sample_size # number of samples to be collected
        self.samples = np.array([self.algorithm(*args) for i in range(self.size)]) # container for the collected samples
        # self.mean = np.mean(self.samples)
        # self.var = np.var(self.samples, ddof = 1) # unbiased estimator
        self.current_value = self.samples[-1]
        return self.samples

    def compare(self, file_path = None, display = True, target_cdf_pts = 100):
        """
        Description:
            Draws cdfs for target and simulation and sets self.ecdf.

        Args:
            file_path: the location where the image file is saved, image file is not saved in case file_path = None (default).
            The plot is displayed on screen if display = True (default).
            target_cdf_pts: number of points used to plot the target cdf.

        Returns:
            figure and axes objects for the generated plot (in this order)
        """
        # compute target mean and variance if not already computed
        self.rv.set_unset_stats(('mean', 'var'))

        # compute and plot simulated cdf
        self.ecdf = ECDF(self.samples)
        fig = plt.figure(figsize = (7,6))
        ax = fig.add_subplot(111)
        ax.plot(self.ecdf.x, self.ecdf.y, label = 'simulation ($\mu$ = {:.4f}, $\sigma^2$ = {:.4f})'.format(self.mean, self.var))

        # fix limits for np.linspace in case rv.a or rv.b is unbounded
        left_lim = self.ecdf.x[1] if np.isinf(self.rv.a) else self.rv.a
        right_lim = self.ecdf.x[-1] if np.isinf(self.rv.b) else self.rv.b

        # plot target cdf
        x = np.linspace(left_lim, right_lim, target_cdf_pts, endpoint = True)
        y = [self.rv.cdf(pt) for pt in x]
        ax.plot(x, y, label = 'target ($\mu$ = {:.4f}, $\sigma^2$ = {:.4f})'.format(self.rv.mean, self.rv.var))

        # write textual info on the plot
        ax.set_title('CDF vs ECDF')
        ax.set_xlabel('x')
        ax.legend()

        # save and display
        if file_path is not None:
            fig.savefig(file_path)
        if display:
            plt.show()
        return fig, ax

    def set_algorithm(self, algorithm, **algorithm_args):
        """
        Description:
            Constructs self.algorithm and sets #self.dimension

        Args:
            algorithm: a function that produces a single sample of target_rv
            algorithm_args: dict of keyword arguments that are passed to algorithm
        """
        # set built-in algorithm for simulation/sampling if possible
        if algorithm == 'inverse':
            algorithm = lambda *args: inverse_transform(*args, self.algorithm_args['inv_cdf'], **self.rv.params) # algorithm_args = {'inv_cdf': -}
        elif algorithm == 'composition':
            algorithm = lambda *args: composition(*args, **self.algorithm_args) # algorithm_args = {'sim_components': -, 'probabilties': -}
        elif algorithm == 'rejection':
            algorithm = lambda *args: rejection(*args, self.rv, **self.algorithm_args) # algorithm_args = {'helper_rv': -, 'ratio_bound': -} (helper_rv must have pdf assigned)
        elif algorithm == 'gamma':
            algorithm = lambda *args: np.random.gamma(*args, **self.rv.params) # to be modified

        self.algorithm = algorithm #lambda *args: algorithm(*args, **algorithm_args)
        """
        # figure out the dimension of the problem
        sample = self.algorithm()
        if np.isscalar(sample):
            self.dimension = 1
        else:
            self.dimension = len(sample)
        """




########################################
# Stochastic Process class definitions #
########################################

class StochasticProcess(object):
        """
        Description:
            This is a class for defining generic stochastic processes.

        Attributes:
            current_path: last generated path of the stochastic process
            sims: Simulation objects for the random variables that make up the stochastic process
            size: number of random variables that make up the stochastic process
            paths: np.array of generated paths (created after first call to generate_paths)

        Methods:
            generate_path: gennerates a single path, sets it to self.current_path
            generate_paths: generates a batch of sample paths
            avg_path: computes the average of self.paths
        """

        def __init__(self, sims):
            """
            Args:
                sims: Simulation objects for random variables X_t that make up the stochastic process
            """
            # assign basic attributes
            self.current_path = None # last generated path of the stochastic process
            self.sims = sims # Simulation objects for the random variables that make up the stochastic process
            self.size = len(sims) # number of the random variables that make up the stochastic process
            # self.dimension = self.sims[0].dimension # dimension of the random variables that make up the stochastic process

        def generate_path(self, *args):
            """
            Description:
                Gennerates a single path, sets it to self.current_path

            Returns:
                self.current_path
            """
            self.current_path = np.array([sim.algorithm(*args) for sim in self.sims]) # last generated path
            return self.current_path

        @ut.timer
        def generate_paths(self, num_paths, *args):
            """
            Description:
                Generates a batch of sample paths

            Args:
                num_paths: number of paths to be generated

            Returns:
                self.paths
            """
            self.paths = np.array([self.generate_path(*args) for i in range(num_paths)]) # container for the generated paths
            return self.paths

        def avg_path(self):
            """
            Description:
                Computes the average of self.paths

            Returns:
                self.avg_path
            """
            self.avg_path = np.average(self.paths, axis = 1) # average of self.paths
            return self.avg_path


class MarkovChain(StochasticProcess):
    """
    Description:
        This is a class defining a Markov process.
        Parent class : StochasticProcess

    Attributes (extra):
        conditional_pdf: p(x_k|x_(k-1)), default = None
        algorithm_args:
    """

    def __init__(self, size, prior, algorithm, conditional_pdf = None, **algorithm_args):
        """
        Args:
            size: number of random variables in the chain
            prior: Simulation object for the first random variable in the chain
            algorithm: algorithm for creating Simulation objects in the chain, first two args: time, past
            conditional_pdf: p(x_k|x_(k-1)), default = None, it's a function of type p(x, condition) (argument names can be anything)
            algorithm_args: dict of keyword arguments that are passed to algorithm
        """
        self.conditional_pdf = conditional_pdf
        self.algorithm_args = algorithm_args
        sims = [prior]
        for i in range(size - 1):
            sims.append(Simulation(algorithm = algorithm, **self.algorithm_args))
        super().__init__(sims)

    def generate_path(self, *args):
        """
        Description:
            Gennerates a single path, sets it to self.current_path

        Returns:
            self.current_path
        """
        self.current_path = [self.sims[0].algorithm()]
        for i, sim in enumerate(self.sims[1:]):
            self.current_path.append(sim.algorithm(i+1, self.current_path[-1])) # last generated path
        self.current_path = np.array(self.current_path)
        return self.current_path


class SPConditional(StochasticProcess):
    """
    Description:
        This is a class defining a StochasticProcess Y_t where the probability density of Y_t is given as a conditional density p(y_t|x_t)
        where X_t is a given StochasticProcess.
        Parent class : StochasticProcess
    """
    def __init__(self, size, algorithm, conditional_pdf = None, **algorithm_args):
        """
        Args:
            conditions: list of Simulation objects that make up X_t
            algorithm: algorithm for creating Simulation objects for Y_t, accepts the condition as a Simulation object in the argument 'condition'
            conditional_pdf: p(y_k|x_k), default = None, it's a function of type p(y, condition) (argument names can be anything)
            algorithm_args: dict of keyword arguments that are passed to algorithm
        """
        self.conditional_pdf = conditional_pdf
        self.algorithm_args = algorithm_args
        sims = []
        for i in range(size):
            sims.append(Simulation(algorithm = algorithm, *self.algorithm_args))
        super().__init__(sims)

    def generate_path(self, conditions):
        """
        Description:
            Gennerates a single path, sets it to self.current_path

        Returns:
            self.current_path
        """
        self.current_path = []
        for i, sim in enumerate(self.sims):
            self.current_path.append(sim.algorithm(i, conditions[i])) # last generated path
        self.current_path = np.array(self.current_path)
        return self.current_path


class DynamicModel(MarkovChain):
    """
    Description:
        This is a class for defining a MarkovChain of the form x_k = f(x_(k-1), z)
        z ~ zero mean noise
        Parent class: MarkovChain
    """
    def __init__(self, size, prior, func, sigma, noise_sim = None, conditional_pdf = None):
        """
        Args:
            size: number of random variables in the chain
            prior: Simulation object for the first random variable in the chain
            func: the function f defining the dynamics, takes 3 arguments time, x and process noise
            sigma: covariance matrix of z, a d-dimensional normal random variable as described in the model
            noise_sim: Simulation object for the process noise
            conditional_pdf: a function of form p(x, condition) = p(x_k|x_(k-1))
        """
        # set parameters for the model
        self.func = func
        self.sigma = sigma
        self.dimension = np.shape(sigma)[0]
        if noise_sim is not None:
            self.noise_sim = noise_sim
        else:
            self.noise_sim = Simulation(algorithm = lambda *args: np.random.multivariate_normal(mean = np.zeros(self.dimension), cov = self.sigma))

        if conditional_pdf is None:
            conditional_pdf = lambda k, x, past: scipy.stats.multivariate_normal.pdf(x, mean = self.func(k, past, np.zeros(self.dimension)), cov = self.sigma)
        # figure out simulation algorithm
        def algorithm(k, past):
            return self.func(k, past, self.noise_sim.algorithm())

        super().__init__(size = size, prior = prior, algorithm = algorithm, conditional_pdf = conditional_pdf)

class MeasurementModel(SPConditional):
    """
    Description:
        This is a class for defining an SPConditional object of the form y_k = f(x_k, z)
        and z ~ zero mean noise
        Parent class: SPConditional
    """
    def __init__(self, size, func, sigma, noise_sim = None, conditional_pdf = None):
        """
        Args:
            size: number of random variables in the chain
            func: function f defining the relationship between hidden state and observation, takes 3 arguments time, x and measurement noise
            sigma: covariance matrix of z, a d-dimensional normal random variable as described in the model
            conditional_pdf: a function of form p(y, condition) = p(y_k|x_k))
        """
        # set parameters for the model
        self.func = func
        self.sigma = sigma
        self.dimension = np.shape(sigma)[0]
        if noise_sim is not None:
            self.noise_sim = noise_sim
        else:
            self.noise_sim = Simulation(algorithm = lambda *args: np.random.multivariate_normal(mean = np.zeros(self.dimension), cov = self.sigma))

        if conditional_pdf is None:
            conditional_pdf = lambda k, y, condition: scipy.stats.multivariate_normal.pdf(y, mean = self.func(k, condition, np.zeros(self.dimension)), cov = self.sigma)
        # figure out simulation algorithm
        def algorithm(k, condition):
            return self.func(k, condition, self.noise_sim.algorithm())

        super().__init__(size = size, algorithm = algorithm, conditional_pdf = conditional_pdf)
