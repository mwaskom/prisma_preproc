"""Heatmap visualization of an fMRI time series for quality control.

Based on approach developed by Jonathan Power and explained here:
https://www.ncbi.nlm.nih.gov/pubmed/27510328

Python implementation by Michael Waskom <mwaskom@nyu.edu>

Released under Revised BSD license.

"""
import numpy as np
from scipy.signal import detrend
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib


class PowerPlot(object):

    def __init__(self, data, wmparc, realign_params=None, smooth_sigma=3,
                 vlim=2, title=None):
        """Heatmap rendering of an fMRI timeseries for quality control.

        The Freesurfer segmentation is used to organize data by different
        components of the brain. The components are organized from top to
        bottom and color-coded as follows:

            - cortex (dark blue)
            - subcortical gray matter (medium blue)
            - cerebellum (light blue)
            - superficial white matter (light red)
            - deep white matter (dark red)
            - ventricles (yellow)

        Instantiating the class will load, preprocess, and plot the data.

        Parameters
        ----------
        data : filename or nibabel image
            4D time series data to plot.
        wmparc : filename or nibabel image
            Freesurfer wmparc image in functional space.
        realign_params : filename or numpy array, optional
            Text file or array of realignment parameters in FSL-style format.
            This means three columns of rotations in radians and then three
            columns of translations in mm. If present, the time series of
            framewise displacements will be shown at the top of the figure.
        smooth_sigma : float,
            Size of the smoothing kernel, in mm, to apply. Smoothing is
            restricted within the mask for each component (cortex, cerebellum,
            etc.). Smoothing reduces white noise and makes global image
            artifacts much more apparent.
        vlim : int
            Colormap limits (will be symmetric) in percent signal change units.
        title : string
            Title to show at the top of the plot.

        Attributes
        ----------
        fig : matplotlib Figure
        axes : dict of matplotlib Axes

        """
        # Load the timeseries data
        if isinstance(data, str):
            img = nib.load(data)
        else:
            img = data
        data = img.get_data().astype(np.float)

        # Load the Freesurfer parcellation
        if isinstance(wmparc, str):
            wmparc = nib.load(wmparc).get_data()
        else:
            wmparc = wmparc.get_data()

        # Use header geometry to convert smoothing sigma from mm to voxels
        sx, sy, sz, _ = img.header.get_zooms()
        voxel_sizes = sx, sy, sz
        if smooth_sigma is not None:
            if smooth_sigma > 0:
                smooth_sigma = np.divide(smooth_sigma, voxel_sizes)
            else:
                smooth_sigma = None

        # Preprocess and segment the data
        masks, brain = self.define_masks(wmparc)
        data[brain] = self.percent_change(data[brain])
        data[brain] = detrend(data[brain])
        data = self.smooth_data(data, masks, smooth_sigma)
        segdata = self.segment_data(data, masks)
        fd = self.framewise_displacement(realign_params)

        # Make the plot
        fig, axes = self.setup_figure()
        self.fig, self.axes = fig, axes
        self.plot_fd(axes["motion"], fd)
        self.plot_data(axes, segdata, vlim)
        if title is not None:
            fig.suptitle(title)

    def percent_change(self, data):
        """Convert to percent signal change over the mean for each voxel."""
        null = data.mean(axis=-1) == 0
        with np.errstate(all="ignore"):
            data /= data.mean(axis=-1, keepdims=True)
        data -= 1
        data *= 100
        data[null] = 0
        return data

    def define_masks(self, wmparc):
        """Create masks for anatomical components using Freesurfer labeling."""
        subgm_ids = [10, 11, 12, 13, 16, 17, 18, 49, 50, 51, 52, 53, 54]
        csf_ids = [4, 43, 31, 63]

        masks = dict(
            cortex=(1000 <= wmparc) & (wmparc < 3000),
            subgm=np.in1d(wmparc.flat, subgm_ids).reshape(wmparc.shape),
            cerbel=(wmparc == 8) | (wmparc == 47),
            supwm=(3000 <= wmparc) & (wmparc < 5000),
            deepwm=(wmparc == 5001) | (wmparc == 5002),
            csf=np.in1d(wmparc.flat, csf_ids).reshape(wmparc.shape),
            )
        brain = np.any(masks.values(), axis=0)
        return masks, brain

    def smooth_data(self, data, masks, sigma):
        """Smooth the 4D image separately within each component."""
        if sigma is None:
            return data

        for comp, mask in masks.items():
            data[mask] = self._smooth_within_mask(data, mask, sigma)

        return data

    def _smooth_within_mask(self, data, mask, sigmas):
        """Smooth each with a Gaussian kernel, restricted to a mask."""
        comp_data = data * np.expand_dims(mask, -1)
        for f in range(comp_data.shape[-1]):
            comp_data[..., f] = gaussian_filter(comp_data[..., f], sigmas)

        smooth_mask = gaussian_filter(mask.astype(float), sigmas)
        with np.errstate(all="ignore"):
            comp_data = comp_data / np.expand_dims(smooth_mask, -1)

        return comp_data[mask]

    def segment_data(self, data, masks):
        """Convert the 4D data image into a set of 2D matrices."""
        segdata = {comp: data[mask] for comp, mask in masks.items()}
        return segdata

    def framewise_displacement(self, realign_params):
        """Compute the time series of framewise displacements."""
        if isinstance(realign_params, str):
            rp = np.loadtxt(realign_params)
        elif isinstance(realign_params, np.ndarray):
            rp = realign_params
        else:
            return None

        r, t = np.hsplit(rp, 2)
        s = r * 50
        ad = np.hstack([s, t])
        rd = np.abs(np.diff(ad, axis=0))
        fd = np.sum(rd, axis=1)
        return fd

    def setup_figure(self):
        """Initialize and organize the matplotlib objects."""
        width, height = 8, 10
        f = plt.figure(figsize=(width, height))

        gs = plt.GridSpec(nrows=2, ncols=2,
                          left=.07, right=.98,
                          bottom=.05, top=.96,
                          wspace=0, hspace=.02,
                          height_ratios=[.1, .9],
                          width_ratios=[.01, .99])

        ax_i = f.add_subplot(gs[1, 1])
        ax_m = f.add_subplot(gs[0, 1], sharex=ax_i)
        ax_c = f.add_subplot(gs[1, 0], sharey=ax_i)

        ax_i.set(xlabel="Volume", yticks=[])
        ax_m.set(ylabel="FD (mm)")
        ax_c.set(xticks=[])

        ax_b = f.add_axes([.035, .35, .0125, .2])

        axes = dict(image=ax_i, motion=ax_m, comp=ax_c, cbar=ax_b)

        return f, axes

    def plot_fd(self, ax, fd):
        """Show a line plot of the framewise displacement data."""
        if fd is None:
            fd = []

        ax.set(ylim=(0, .5))
        ax.plot(np.arange(1, len(fd) + 1), fd, lw=1.5, color=".15")
        ax.set(ylabel="FD (mm)", ylim=(0, None))
        for label in ax.get_xticklabels():
            label.set_visible(False)

    def plot_data(self, axes, segdata, vlim):
        """Draw the elements corresponding to the image data."""
        components = ["cortex", "subgm", "cerbel", "supwm", "deepwm", "csf"]
        plot_data = np.vstack([segdata[comp] for comp in components])

        axes["image"].imshow(plot_data, cmap="gray", vmin=-vlim, vmax=vlim,
                             aspect="auto", rasterized=True)

        sizes = [len(segdata[comp]) for comp in components]
        cum_sizes = np.cumsum(sizes)

        for y in cum_sizes[:-1]:
            axes["image"].axhline(y, c="w", lw=1)

        comp_ids = np.vstack([
            np.full((len(segdata[comp]), 1), i, dtype=np.int)
            for i, comp in enumerate(components)
        ])
        comp_colors = [u'#00035b', u'#3b638c', u'#5a86ad',
                       u'#b9484e', u'#8c000f', u'#fbdd7e']
        comp_cmap = mpl.colors.ListedColormap(comp_colors)
        axes["comp"].imshow(comp_ids,
                            vmin=0, vmax=len(components) - 1,
                            aspect="auto", rasterized=True,
                            cmap=comp_cmap)

        xx = np.expand_dims(np.linspace(1, 0, 100), -1)
        ax = axes["cbar"]
        ax.imshow(xx, aspect="auto", cmap="gray")
        ax.set(xticks=[], yticks=[], ylabel="Percent signal change")
        ax.text(0, -2, "$+${}".format(vlim),
                ha="center", va="bottom", clip_on=False)
        ax.text(0, 103, "$-${}".format(vlim),
                ha="center", va="top", clip_on=False)
