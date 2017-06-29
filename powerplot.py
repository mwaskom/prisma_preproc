import time
import numpy as np
from scipy.signal import detrend
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib


class PowerPlot(object):

    def __init__(self, data, wmparc, realign_params=None, smooth_sigma=3):

        if isinstance(data, str):
            img = nib.load(data)
        else:
            img = data
        data = img.get_data().astype(np.float)

        if isinstance(wmparc, str):
            wmparc = nib.load(wmparc).get_data()
        else:
            wmparc = wmparc.get_data()

        sx, sy, sz, _ = img.header.get_zooms()
        voxel_sizes = sx, sy, sz
        if smooth_sigma is not None:
            if smooth_sigma > 0:
                smooth_sigma = np.divide(smooth_sigma, voxel_sizes)
            else:
                smooth_sigma = None

        components, brain = self.define_components(wmparc)
        data[brain] = self.percent_change(data[brain])
        data[brain] = detrend(data[brain])
        data = self.smooth_data(data, components, smooth_sigma)
        segdata = self.segment_data(data, components)
        fd = self.framewise_displacement(realign_params)

        f, axes = self.setup_figure()
        self.f, self.axes = f, axes

        self.plot_fd(axes["motion"], fd)
        self.plot_data(axes, segdata)

    def percent_change(self, data):

        null = data.mean(axis=-1) == 0
        with np.errstate(all="ignore"):
            data /= data.mean(axis=-1, keepdims=True)
        data -= 1
        data *= 100
        data[null] = 0
        return data

    def define_components(self, wmparc):

        subgm_ids = [10, 11, 12, 13, 16, 17, 18, 49, 50, 51, 52, 53, 54]
        csf_ids = [4, 43, 31, 63]

        components = dict(
            cortex=(1000 <= wmparc) & (wmparc < 3000),
            subgm=np.in1d(wmparc.flat, subgm_ids).reshape(wmparc.shape),
            cerbel=(wmparc == 8) | (wmparc == 47),
            supwm=(3000 <= wmparc) & (wmparc < 5000),
            deepwm=(wmparc == 5001) | (wmparc == 5002),
            csf=np.in1d(wmparc.flat, csf_ids).reshape(wmparc.shape),
            )
        brain = np.any(components.values(), axis=0)
        return components, brain

    def smooth_data(self, data, components, sigma):

        if sigma is None:
            return data

        for comp, mask in components.items():
            data[mask] = self._smooth_within_mask(data, mask, sigma)

        return data

    def _smooth_within_mask(self, data, mask, sigmas):

        comp_data = data * np.expand_dims(mask, -1)
        for f in range(comp_data.shape[-1]):
            comp_data[..., f] = gaussian_filter(comp_data[..., f], sigmas)

        smooth_mask = gaussian_filter(mask.astype(float), sigmas)
        with np.errstate(all="ignore"):
            comp_data = comp_data / np.expand_dims(smooth_mask, -1)

        return comp_data[mask]

    def segment_data(self, data, components):

        segdata = {comp: data[mask] for comp, mask in components.items()}
        return segdata

    def framewise_displacement(self, realign_params):

        if realign_params is None:
            return None

        rp = np.loadtxt(realign_params)
        r, t = np.hsplit(rp, 2)
        s = r * 50
        ad = np.hstack([s, t])
        rd = np.abs(np.diff(ad, axis=0))
        fd = np.sum(rd, axis=1)
        return fd

    def setup_figure(self):

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
        ax_m.set_ylabel("FD (mm)")
        ax_c.set(xticks=[])

        ax_b = f.add_axes([.035, .35, .0125, .2])

        axes = dict(image=ax_i, motion=ax_m, components=ax_c, cbar=ax_b)

        return f, axes

    def plot_fd(self, ax, fd):

        if fd is None:
            pass

        ax.set(ylim=(0, .5))
        ax.plot(np.arange(1, len(fd) + 1), fd)
        ax.set(ylabel="FD (mm)")
        ax.set(ylim=(0, None))

    def plot_data(self, axes, segdata):

        components = ["cortex", "subgm", "cerbel", "supwm", "deepwm", "csf"]
        plot_data = np.vstack([segdata[comp] for comp in components])

        axes["image"].imshow(plot_data, cmap="gray", vmin=-2, vmax=2,
                             aspect="auto", rasterized=True)

        sizes = [len(segdata[comp]) for comp in components]
        cum_sizes = np.cumsum(sizes)

        for y in cum_sizes[:-1]:
            axes["image"].axhline(y, c="w", lw=1)

        comp_ids = np.vstack([
            np.full((len(segdata[comp]), 1), i, dtype=np.int)
            for i, comp in enumerate(components)
        ])
        axes["components"].imshow(comp_ids, aspect="auto", cmap="viridis",
                                  rasterized=True)

        xx = np.linspace(1, 0, 100)[:, np.newaxis]
        ax = axes["cbar"]
        ax.imshow(xx, aspect="auto", cmap="gray")
        ax.set(xticks=[], yticks=[], ylabel="Percent signal change")
        ax.text(0, -2, "+2", ha="center", va="bottom", clip_on=False)
        ax.text(0, 103, "-2", ha="center", va="top", clip_on=False)
