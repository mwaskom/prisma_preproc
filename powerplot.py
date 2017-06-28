
import numpy as np
from scipy.signal import detrend
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import nibabel as nib


class PowerPlot(object):

    def __init__(self, data, wmparc, realign_params=None, smooth_sigma=2):

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

        voxel_sizes = np.array(img.header.get_zooms()[:3])

        print("Detrending and converting data to percent change")
        data = self.percent_change(data)
        self.data = data = detrend(data)
        print("Defining image components")
        self.components = components = self.define_components(wmparc)
        print("Smoothing data within-component")
        self.data = data = self.smooth_data(data, components,
                                            smooth_sigma, voxel_sizes)
        print("Extracting data from each component")
        self.segdata = segdata = self.segment_data(data, components)

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

    def smooth_data(self, data, components, sigma, voxel_sizes):

        if sigma is None or sigma == 0:
            return data
        else:
            sigmas = sigma / voxel_sizes

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
