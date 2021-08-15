import numpy as np
from tables import description
from tables.description import Description
import ipywidgets as widgets 
from IPython.display import display 
import matplotlib.pyplot as plt
import tables
import pandas as pd
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AsmlWidget:
    """
    Description:
        widget for visualizing assimilation
    Attributes:
        asml_file_path: path to assimilation file
        hidden_file_path: path to true trajectory
        attractor_path: path to attractor points
        attractor: array of attractor points
        data: handle to assimilation file
    """
    def __init__(self, asml_file_path, dims=[0, 1], hidden_file_path=None, attractor_path=None, fig_size=(10, 5), max_attractor_pts=3000):
        self.asml_file_path = asml_file_path
        self.dims = dims
        self.attractor_path = attractor_path
        self.hidden_file_path = hidden_file_path
        self.fig_size = fig_size
        self.data = tables.open_file(self.asml_file_path, 'r')
        self.observation = np.array(self.data.root.observation.read().tolist())
        self.num_steps = len(self.observation)
        self.l2_err = np.zeros(self.num_steps)
        if hidden_file_path is not None:
            self.hidden_path = self.get_hidden_path()
        if attractor_path is not None:
            self.attractor = self.get_attractor()
            if len(self.attractor) > max_attractor_pts:
                self.attractor = self.attractor[np.random.choice(len(self.attractor), size=max_attractor_pts, replace=False)]
        self.asml_step_slider = widgets.IntSlider(value=0, min=0, max=self.num_steps-1, step=1)
        widgets.interact(self.ensemble_plot, step=self.asml_step_slider)

    def ensemble_plot(self, step):
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(121)
        posterior = self.get_ensemble(step)
        if self.attractor_path is not None:
            ax.scatter(self.attractor[:, 0], self.attractor[:, 1], c='green', s=100, label='attractor')
        ax.scatter(posterior[:, 0], posterior[:, 1], c='blue', label='posterior')
        ax.scatter(self.hidden_path[step, 0], self.hidden_path[step, 1], c='pink', label='true_state')
        self.l2_err[step] = np.linalg.norm(np.mean(posterior, axis=0) - self.hidden_path[step])
        ax_l2 = fig.add_subplot(122)
        ax_l2.set_title('L2 error = {:.2f}'.format(self.l2_err[step]))
        ax_l2.plot(range(self.num_steps), self.l2_err)
        ax_l2.scatter(step, self.l2_err[step], s=30, c='red')
        ax.legend()

    def get_ensemble(self, step):
        return np.array(getattr(self.data.root.particles, 'time_' + str(step)).read().tolist())[:, self.dims]

    def get_attractor(self):
        return np.genfromtxt(self.attractor_path, delimiter=',')[:, self.dims]

    def get_hidden_path(self):
        return np.genfromtxt(self.hidden_file_path, delimiter=',')[:self.num_steps, self.dims]
        

class PCShapeWidget:
    """
    Description:
        widget for visualizing point cloud shape
    Attributes:
        asml_file_path: path to assimilation file
        data: handle to assimilation file
    """
    def __init__(self, asml_file_path, dims=[0, 1], nb_type='jupyter'):#, fig_size=(10, 5)):
        self.asml_file_path = asml_file_path
        self.dims = dims
        self.nb_type = nb_type
        #self.fig_size = fig_size
        self.data = tables.open_file(self.asml_file_path, 'r')
        self.observation = np.array(self.data.root.observation.read().tolist())
        self.num_steps = len(self.observation)
        self.asml_step_slider = widgets.IntSlider(value=0, min=0, max=self.num_steps-1, step=1, description='step')
        self.button = widgets.Button(description='compute and plot')
        
        if len(dims) > 2:
            self.fig = go.FigureWidget(make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]], subplot_titles=['posterior', 'Gaussian counterpart']))
            self.button.on_click(self.plot_hull_3d)
        else:
            self.fig = go.FigureWidget(make_subplots(rows=1, cols=2, subplot_titles=['posterior', 'Gaussian counterpart']))
            self.button.on_click(self.plot_hull_2d)
        if self.nb_type == 'colab':
            display(self.button, self.asml_step_slider)
        else:
            display(self.button, self.asml_step_slider, self.fig)
    
    def plot_hull_3d(self, button):
        self.fig.data = []
        self.fig.layout.title = 'step = {}'.format(self.asml_step_slider.value)
        X = self.get_ensemble(self.asml_step_slider.value)
        hull = X[ConvexHull(X).vertices]
        self.fig.add_trace(go.Mesh3d(x=hull[:, 0], y=hull[:, 1], z=hull[:, 2], color="blue", opacity=0.4, alphahull=0), row=1, col=1)
        self.fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers'), row=1, col=1)
        mean = np.mean(X, axis=0)
        X = X - mean
        cov = (X.T @ X) / (X.shape[0]-1)
        X = np.random.multivariate_normal(mean, cov, size=X.shape[0])
        hull = X[ConvexHull(X).vertices]
        self.fig.add_trace(go.Mesh3d(x=hull[:, 0], y=hull[:, 1], z=hull[:, 2], color="pink", opacity=0.4, alphahull=0), row=1, col=2)
        self.fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers'), row=1, col=2)
        if self.nb_type == 'colab':
            self.fig.show()

    def plot_hull_2d(self, button):
        self.fig.data = []
        self.fig.layout.title = 'step = {}'.format(self.asml_step_slider.value)
        X = self.get_ensemble(self.asml_step_slider.value)
        hull = X[ConvexHull(X).vertices]
        hull = np.append(hull, [hull[0]], axis=0)
        self.fig.add_trace(go.Scatter(x=hull[:, 0], y=hull[:, 1], fillcolor="blue", opacity=0.4, fill='toself'), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers'), row=1, col=1)
        mean = np.mean(X, axis=0)
        X = X - mean
        cov = (X.T @ X) / (X.shape[0]-1)
        X = np.random.multivariate_normal(mean, cov, size=X.shape[0])
        hull = X[ConvexHull(X).vertices]
        hull = np.append(hull, [hull[0]], axis=0)
        self.fig.add_trace(go.Scatter(x=hull[:, 0], y=hull[:, 1], fillcolor="pink", opacity=0.4, fill='toself'), row=1, col=2)
        self.fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers'), row=1, col=2)
        if self.nb_type == 'colab':
            self.fig.show()

    def get_ensemble(self, step):
        return np.array(getattr(self.data.root.particles, 'time_' + str(step)).read().tolist())[:, self.dims]