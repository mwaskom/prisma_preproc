
import numpy as np
from scipy.signal import detrend
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import nibabel as nib


class PowerPlot(object):

    def __init__(self, data, wmparc, realign_params=None,
                 smooth_sigma=3, random_seed=0):

        print("Loading data")
        if isinstance(data, str):
            img = nib.load(data)
        else:
            img = data
        self.orig_data = data = img.get_data().astype(np.float).copy()

        if isinstance(wmparc, str):
            wmparc = nib.load(wmparc).get_data()
        else:
            wmparc = wmparc.get_data()

        sx, sy, sz, tr = img.header.get_zooms()
        self.voxel_sizes = sx, sy, sz
        self.tr = tr
        self.ntp = data.shape[-1]

        print("Converting data to percent signal change")
        data = self.percent_change(data)
        print("Detrending data")
        self.data = data = detrend(data)
        print("Defining image components")
        self.components = components = self.define_components(wmparc)
        print("Smoothing data within-component")
        self.data = data = self.smooth_data(data, components, smooth_sigma)
        print("Extracting data from each component")
        self.segdata = segdata = self.segment_data(data, components)
        print("Computing framewise displacement")
        self.fd = fd = self.framewise_displacement(realign_params)

        print("Plotting")
        f, axes = self.setup_figure()
        self.f, self.axes = f, axes

        self.plot_fd(axes[0], fd)
        self.plot_data(axes[1], segdata)
        f.tight_layout(h_pad=0, w_pad=0)

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
        return components

    def smooth_data(self, data, components, sigma):

        if sigma is None or sigma == 0:
            return data
        else:
            sigmas = np.divide(sigma, self.voxel_sizes)

        for comp, mask in components.items():
            data[mask] = self._smooth_within_mask(data, mask, sigmas)

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

        f = plt.figure(figsize=(width, height)

        f, axes = plt.subplots(2, figsize=(width, height), sharex=True,
                               gridspec_kw=dict(height_ratios=(.1, .9)))

        return f, axes

    def plot_fd(self, ax, fd):

        if fd is None:
            pass

        ax.plot(fd)
        ax.set(ylabel="FD (mm)")
        if fd.max() < .2:
            ax.set(ylim=(0, .2))
        else:
            ax.set(ylim=(0, None))

    def plot_data(self, ax, segdata):

        rs = np.random.RandomState(0)

        plot_data = np.vstack([
            self._subsample(segdata["cortex"], 600, rs),
            self._subsample(segdata["subgm"], 200, rs),
            self._subsample(segdata["cerbel"], 200, rs),
            self._subsample(segdata["supwm"], 400, rs),
            self._subsample(segdata["deepwm"], 400, rs),
            self._subsample(segdata["csf"], 200, rs),
            ])

        ax.imshow(plot_data, aspect="auto", cmap="gray", vmin=-2, vmax=2)

        ax.axhline(600, c="w")
        ax.axhline(800, c="w")
        ax.axhline(1000, c="w")
        ax.axhline(1400, c="w")
        ax.axhline(1800, c="w")

        ax.set(yticks=[])

    def _subsample(self, data, n, rs):

        return rs.permutation(data)[:n]
