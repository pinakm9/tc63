import numpy as np
import utility as ut
import  matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from PIL import Image
import re
import os

class SignalPlotter(object):
    """
    Description:
        A class for plotting signals. Signal is a timeseries whose every can be a scalar or a vector (matrices and higher rank tensors are not supported).

    Attributes:
        signal:
        start_time:
        time_step:
        algorithm:

    Methods:
        plot_signals:
    """
    def __init__(self, signals = [], dimension = None, start_time = 0.0, time_step = 1.0):
        """
        Args:
            signals: signals to be processed
            start_time: time at first obeservation, default = 0.0
            time_step: time step between consecutive observations, default = 1.0
        """
        # assign basic attributes
        self.signals = signals
        self.start_time = start_time
        self.time_step = time_step
        self.algorithm = None
        self.processed = []

        # figure out the dimension of the problem
        if dimension is None:
            if len(np.shape(signals[0])) == 2:
                self.dimension = np.shape(signals[0])[1]
            else:
                self.dimension = 1
        else:
            self.dimension = dimension

    def plot_signals(self, labels = [], styles = [{'linestyle':'solid'}, {'marker':'o'}, {'marker':'^'}],\
                    plt_fns = ['plot', 'scatter', 'scatter'],  colors = ['red', 'green', 'blue'],\
                    max_pts = 100, fig_size = (7,6), time_unit = 'second', coords_to_plot = [],\
                    show = False, file_path = None, title = None):
        """
        Description:
            Plots observed and processed signals depending on the dimension of the problem

        Args:
            labels: identifiers for the signals
            styles: line styles for signals
            max_pts: Maximum number of points (default = 100) to be plotted for each signal
            fig_size: size of the plot as a tuple (unit of length as in matplotlib standard)
            time_unit: unit of time to be displayed in x-label for 1-dimensional problems
            coords_to_plot: list of coordinates to plot, default is [] for which all coordinates are plotted (together in case dimension < 4 and separately otherwise)

        Returns:
            figure and axes objects created (axes is a list of matplotlib axes in case coords_to_plot is not empty)
        """
        # prepare a figure
        fig = plt.figure(figsize = fig_size)

        # fix styles, labels, plt_fns and colors if their lengths are not adequate
        if len(self.signals) > len(styles):
            styles += [{'marker': 'x'}]*(len(self.signals) - len(styles))
        if len(self.signals) > len(labels):
            labels += ['']*(len(self.signals) - len(labels))
        if len(self.signals) > len(plt_fns):
            plt_fns += ['scatter']*(len(self.signals) - len(plt_fns))
        if len(self.signals) > len(colors):
            colors += ['blue']*(len(self.signals) - len(colors))

        # plot self.signals against time
        if self.dimension == 1 and coords_to_plot == []:
            ax = fig.add_subplot(111)
            t = np.linspace(self.start_time, self.start_time + (len(self.signals[0])-1)*self.time_step, num = min(max_pts, len(self.signals[0])))
            for i, signal in enumerate(self.signals):
                signal = ut.Picker(signal).equidistant(objs_to_pick = max_pts)
                getattr(ax, plt_fns[i])(t, signal, label = labels[i], color = colors[i], **styles[i])
            ax.set(xlabel = 'time({})'.format(time_unit))
            ax.legend()

        # plot all coordinatres of the self.signals together, time is not shown
        elif self.dimension == 2 and coords_to_plot == []:
            ax = fig.add_subplot(111)
            for i, signal in enumerate(self.signals):
                signal = ut.Picker(signal).equidistant(objs_to_pick = max_pts)
                getattr(ax, plt_fns[i])(signal[:, 0], signal[:, 1], label = labels[i], color = colors[i], **styles[i])
            ax.legend()

        # plot all coordinatres of the self.signals together, time is not shown
        elif self.dimension == 3 and coords_to_plot == []:
            ax = fig.add_subplot(111, projection='3d')
            for i, signal in enumerate(self.signals):
                signal = ut.Picker(signal).equidistant(objs_to_pick = max_pts)
                getattr(ax, plt_fns[i])(signal[:, 0], signal[:, 1], signal[:, 2], label = labels[i], color = colors[i], **styles[i])
            ax.legend()

        # plot the required coordinates separately against time
        elif self.dimension > 3 or coords_to_plot != []:
            ax, num_rows = [], len(coords_to_plot)
            t = np.linspace(self.start_time, self.start_time + (len(self.signals[0])-1)*self.time_step, min(max_pts, len(self.signals[0])))
            for i in range(num_rows):
                ax.append(fig.add_subplot(num_rows, 1, i+1))
                for j, signal in enumerate(self.signals):
                    try:
                        signal = ut.Picker(signal[:, coords_to_plot[i]]).equidistant(objs_to_pick = max_pts)
                    except:
                        continue
                    getattr(ax[i], plt_fns[j])(t, signal, label = labels[j], color = colors[j], **styles[j])
                    ax[i].set(ylabel = 'dimension {}'.format(coords_to_plot[i] + 1))
                    ax[i].yaxis.set_label_position('right')
                ax[i].legend()
            fig.text(0.5, 0.05, 'time({})'.format(time_unit), ha='center', va='center')

        if title is not None:
            plt.title(title)

        if file_path is not None:
                plt.savefig(fname = file_path)

        if show is True:
            plt.show()
        #else:
            #print("file_path was not specified. So the image file was not saved.")
        
        return fig, ax


class EnsemblePlotter:
    """
    Description:
        Plots evolution of ensembles
    """
    def __init__(self, fig_size = (10, 10), pt_size = 1, size_factor = 5, dpi = 300):
        self.fig_size = fig_size
        self.pt_size = pt_size
        self.size_factor = size_factor

        # generate figure for the plot
        self.fig = plt.figure(figsize = self.fig_size, dpi = dpi)

    @ut.timer
    def plot_weighted_ensembles_2D(self, ensembles, weights, ens_labels, colors, file_path, alpha = 0.5, log_size = False, weight_histogram = True,\
                                   log_weight = False, extra_data = [], extra_plt_fns = [], extra_styles = [], extra_labels = [], extra_colors = []):
        """
        Description:
            Plots a 2D weighted ensemble
        Args:
            ensemble: ensemble to be plotted
            weights: weights of particles
            file_path: path where the plot is to be saved
            ax: axes of the plot
            color: color of the points plotted
        """
        # check the number of ensembles to be plotted
        if len(np.array(ensembles).shape) < 3:
            ensembles = [ensembles]
            weights = [weights]
        # normalize weights
        l = len(ensembles)
        for k in range(l):
            weights[k] = np.array(weights[k])
            weights[k] /= weights[k].sum()
        log_weights = np.log(weights)
        log_weights_max = np.amax(log_weights)
        weights_max = np.amax(weights)
        self.fig.clf()
        plt.figure(self.fig.number)
        ax = plt.subplot2grid((self.size_factor*l, self.size_factor*l), (0, 0), rowspan = self.size_factor*l, colspan = (self.size_factor - 1)*l)
        # plot ensembles
        for k, ensemble in enumerate(ensembles):
            # plot weighted points
            for i, pt in enumerate(ensemble):
                if log_size:
                    sz = self.pt_size * (log_weights_max / log_weights[k][i])
                else:
                    sz = self.pt_size * (weights[k][i] / weights_max)
                ax.scatter(pt[0], pt[1], s = sz, color = colors[k], label = ens_labels[k] if i == 0 else None, alpha = alpha)
            # plot weight histogram if needed
            if weight_histogram:
                if log_weight:
                    w = log_weights[k]
                else:
                    w = weights[k]
                w = w[w > -1e300]
                h_ax = plt.subplot2grid((self.size_factor*l, self.size_factor*l), (self.size_factor*k, self.size_factor*l-2), rowspan = l, colspan = l)
                h_ax.hist(w, label = ens_labels[k])
                h_ax.yaxis.tick_right()
                h_ax.legend()
        # plot extra data if needed
        for k, ed in enumerate(extra_data):
            ed = np.array(ed)
            if len(ed.shape) < 2:
                getattr(ax, extra_plt_fns[k])(ed[0], ed[1], color = extra_colors[k], label = extra_labels[k], **extra_styles[k])
            elif ed.shape[0] > 2:
                getattr(ax, extra_plt_fns[k])(ed[:, 0], ed[:, 1], color = extra_colors[k], label = extra_labels[k], **extra_styles[k])
            else:
                getattr(ax, extra_plt_fns[k])(ed[0, :], ed[1, :], color = extra_colors[k], label = extra_labels[k], **extra_styles[k])
        # save and clear figure for reuse
        ax.legend()
        plt.savefig(file_path)

    @ut.timer
    def stich(self, folder, img_prefix, pdf_name, clean_up = True, resolution = 300):
        pages = []
        imgs = []
        if folder.endswith('/'):
            folder = folder[:-1]
        for img in os.listdir(folder):
            if img.startswith(img_prefix):
                pages.append(int(re.findall(r'[0-9]+', img)[-1]))
                im_path = folder + '/' + img
                im = Image.open(im_path)
                rgb_im = Image.new('RGB', im.size, (255, 255, 255))  # white background
                rgb_im.paste(im, mask=im.split()[3])
                imgs.append(rgb_im)
                if clean_up:
                    os.remove(im_path)
        imgs = [imgs[i] for i in np.array(pages).argsort()]
        imgs[0].save(folder + '/' + pdf_name, "PDF", resolution = resolution, save_all = True, append_images = imgs[1:])




def random_color(as_str=True, alpha=0.5):
	rgb = [random.randint(0,255),
		   random.randint(0,255),
		   random.randint(0,255)]
	if as_str:
		return "rgba"+str(tuple(rgb+[alpha]))
	else:
		# Normalize & listify
		return list(np.array(rgb)/255) + [alpha]


def plot_ensemble_trajectory(ensemble_trajectory, ax = None, fig_size = (10, 10), color = 'blue', mean = False, show = True, saveas = None):
    """
    Description: Plots a trajectory of ensembles

    Args:
        ensemble_trajectory: list of ensembles
        ax: axes object for creating the plot
        fig_size: size of the image
        color: color of scatter plot
        show: boolean flag for displaying the generated image
        saveas: file path for the image, default = None in which case the plot won't be saved
    """
    if ax is None:
        fig = plt.figure(figsize = fig_size)
        ax = fig.add_subplot(111)
    for ensemble in ensemble_trajectory:
        ax.scatter(ensemble[0, :], ensemble[1, :], color = random_color(as_str = False))
    if mean:
        x = [np.average(e[0, :]) for e in ensemble_trajectory]
        y = [np.average(e[1, :]) for e in ensemble_trajectory]
        ax.scatter(x, y)
    if show:
        plt.show()
    if saveas is not None:
        plt.savefig(fname = saveas)
    return ax


def plot_ensemble_trajectories(ensemble_trajectories, fig_size = (10, 10), colors = None, show = True, saveas = None):
    """
    Description: Plots a trajectory of ensembles

    Args:
        ensemble_trajectory: list of ensembles
        fig_size: size of the image
        colors: colors of scatter plots
        show: boolean flag for displaying the generated image
        saveas: file path for the image, default = None in which case the plot won't be saved
    """
    i = 0
    ax = None
    if colors is None:
        colors = [ut.random_color(as_str = False, alpha = 1.0) for et in ensemble_trajectories]
        print(colors)
    while i < len(ensemble_trajectories):
        ax = plot_ensemble_trajectory(ensemble_trajectories[i], ax = ax, fig_size = fig_size, color = colors[i], show = False, saveas = None)
        i += 1
    if show:
        plt.show()
    if saveas is not None:
        plt.savefig(fname = saveas)
    return ax


@ut.timer
def plot_frames(et_list, folder, labels, color_list = ['red', 'green', 'blue'], fig_size = (7, 7), dpi = 300):
    """
    Description:
        Plots frames of multiple ensembles
    """
    time_span = len(et_list[0])
    fig = plt.figure(figsize = fig_size, dpi = dpi)
    for t in range(time_span):
        print('Working on frame {} ...'.format(t))
        ax = fig.add_subplot(111)
        for i, et in enumerate(et_list):
            if len(et.shape) > 2:
                ax.scatter(et[t][0, :], et[t][1, :], color = color_list[i], label = labels[i])
            else:
                ax.scatter(et[t][0], et[t][1], color = color_list[i], label = labels[i])
        plt.legend()
        if not folder.endswith('/'):
            folder += '/'
        plt.savefig(folder + 'frame_{}.png'.format(t))
        plt.clf()


@ut.timer
def im2pdf(im_folder, im_prefix, num_im, im_format, pdf_name):
    """
    Description:
        Creates a pdf from a list of images

    Args:
        im_folder: folder that contains the images
        im_prefix: the prefix that the image names start with
        im_format: image file extension
        num_im: number of images to join
        pdf_name: filename(path) for the pdf to be created
    """
    im_list = []
    if not im_folder.endswith('/'):
        im_folder += '/'
    if not im_format.startswith('.'):
        im_format = '.' + im_format
    for i in range(num_im):
        im_name = im_folder + im_prefix + str(i) + im_format
        im = Image.open(im_name)
        rgb_im = Image.new('RGB', im.size, (255, 255, 255))  # white background
        rgb_im.paste(im, mask=im.split()[3])
        im_list.append(rgb_im)
    im_list[0].save(pdf_name, "PDF" ,resolution=300.0, save_all=True, append_images=im_list[1:])
