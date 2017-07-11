from __future__ import division
import os
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, convolve, label
import nibabel as nib
from nipype import (Workflow, Node, MapNode,
                    IdentityInterface, SelectFiles, DataSink)
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    TraitedSpec, File, traits)
from nipype.interfaces import fsl, freesurfer as fs


data_dir = os.environ["SUBJECTS_DIR"]

# --- Define data inputs

subject_source = Node(IdentityInterface(["subject"]),
                      name="subject_source",
                      iterables=("subject", ["rk"]))


templates = dict(t1w_files="{subject}/anat/T1w_*.nii.gz",
                 t2w_files="{subject}/anat/T2w_*.nii.gz")
file_input = Node(SelectFiles(templates,
                              base_directory=data_dir),
                  "file_input")

reorient_t1w = MapNode(fsl.Reorient2Std(), "in_file", "reorient_t1w")
reorient_t2w = MapNode(fsl.Reorient2Std(), "in_file", "reorient_t2w")

# --- Register and average the repetitions of the anatomical volumes

register_t1w = Node(fs.RobustTemplate(initial_timepoint=1,
                                      fixed_timepoint=True,
                                      average_metric="median",
                                      auto_detect_sensitivity=True,
                                      intensity_scaling=True,
                                      no_iteration=True),
                    "register_t1w")

register_t2w = register_t1w.clone("register_t2w")

register_t1w.inputs.out_file = "T1w.nii.gz"
register_t2w.inputs.out_file = "T2w.nii.gz"


# --- Register the T1w volume to the MNI template

mni_template = fsl.Info.standard_image("MNI152_T1_2mm.nii.gz")
mni_mask = fsl.Info.standard_image("MNI152_T1_2mm_brain_mask_dil.nii.gz")

normalize_linear = Node(fsl.FLIRT(dof=12,
                                  reference=mni_template,
                                  interp="spline"),
                        "normalize_linear")

normalize_nonlinear = Node(fsl.FNIRT(config_file="T1_2_MNI152_2mm",
                                     ref_file=mni_template,
                                     refmask_file=mni_mask,
                                     fieldcoeff_file=True),
                           "normalize_nonlinear")

# --- Invert the warpfield and bring the MNI mask into native space

invert_warp = Node(fsl.InvWarp(), "invert_warp")

warp_mask = Node(fsl.ApplyWarp(in_file=mni_mask, interp="nn"),
                 "warp_mask")

fill_mask = Node(fsl.ImageMaths(op_string="-fillh"), "fill_mask")

# --- Register the T2w to the T1w volume

parallelization = dict(OMP_NUM_THREADS="32")
robust_args = "--cost ROBENT --entradius 2 --entcorrection"
register_between = Node(fs.RobustRegister(registered_file=True,
                                          est_int_scale=True,
                                          auto_sens=True,
                                          args=robust_args,
                                          environ=parallelization
                                          ),
                        "register_between")

# TODO add fast/less accurate FLIRT based registration as an option
# register_between = Node(fsl.FLIRT(dof=6, interp="spline"),
#                         "register_between")

# --- Compute the bias field


class BiasCorrectInput(BaseInterfaceInputSpec):

    t1w_file = File(exists=True)
    t2w_file = File(exists=True)
    mask_file = File(exists=True)
    smooth_sigma = traits.Float(5, usedefault=True)


class BiasCorrectOutput(TraitedSpec):

    t1w_file = File(exists=True)
    t2w_file = File(exists=True)
    t1w_brain_file = File(exists=True)
    t2w_brain_file = File(exists=True)
    bias_file = File(exists=True)
    bias_orig_file = File(exists=True)


class BiasCorrect(BaseInterface):

    input_spec = BiasCorrectInput
    output_spec = BiasCorrectOutput

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):

        old_settings = np.seterr(all="ignore")

        self._results = dict()

        # Load the input data
        t1w_img = nib.load(self.inputs.t1w_file)
        t2w_img = nib.load(self.inputs.t2w_file)
        t1w_data, t2w_data = t1w_img.get_data(), t2w_img.get_data()
        brain_mask = nib.load(self.inputs.mask_file).get_data()

        # Compute, normalize, and save the bias field
        bias_data = np.sqrt(np.abs(t1w_data * t2w_data)) * brain_mask
        mean_bias = bias_data[brain_mask.astype(bool)].mean()
        bias_data = (bias_data / mean_bias)
        self.save_output(bias_data, t1w_img,
                         "bias_orig_file", "bias_field_orig.nii.gz")

        # Define smoothing sigma in voxel units
        voxel_res = t1w_img.header.get_zooms()
        voxel_sigma = np.divide(self.inputs.smooth_sigma, voxel_res)

        # Do an initial rough smoothing within in the brain
        smoothed_bias = gaussian_filter(bias_data, voxel_sigma)
        smoothed_mask = gaussian_filter(brain_mask, voxel_sigma)
        smoothed_bias /= smoothed_mask
        smoothed_bias[brain_mask == 0] = 0

        # Generate a tighter mask of only brain tissue
        signal_map = bias_data / smoothed_bias
        signal_mean = signal_map[brain_mask.astype(bool)].mean()
        signal_std = signal_map[brain_mask.astype(bool)].std()
        signal_thresh = signal_mean - (.5 * signal_std)
        high_signal = binary_erosion(signal_map > signal_thresh)
        mask_components, _ = label(high_signal)
        labels = mask_components[mask_components > 0]
        biggest_object = np.argmax(np.bincount(labels))
        tight_mask = mask_components == biggest_object

        # Dilate the bias field estimate to cover full FOV
        weights = np.full((3, 3, 3), 1 / 27)
        bias_data_dil = (bias_data * tight_mask).copy()
        iter_mask = bias_data_dil > 0
        while not iter_mask.all():
            filt = convolve(bias_data_dil, weights)
            norm = convolve(iter_mask.astype(np.float), weights)
            next_dil = filt / norm
            next_dil[norm == 0] = 0
            bias_data_dil[~tight_mask] = next_dil[~tight_mask]
            iter_mask = bias_data_dil > 0

        # Smooth the full-fov bias estimate
        bias_data_dil = gaussian_filter(bias_data_dil, voxel_sigma)

        # Save the final bias field image
        self.save_output(bias_data_dil, t1w_img,
                         "bias_file", "bias_field.nii.gz")

        # Bias-correct and save the T1w image
        t1w_corrected = t1w_data / bias_data_dil
        self.save_output(t1w_corrected, t1w_img, "t1w_file", "T1w.nii.gz")
        t1w_corrected_brain = t1w_corrected * brain_mask
        self.save_output(t1w_corrected_brain, t1w_img,
                         "t1w_brain_file", "T1w_brain.nii.gz")

        # Bias-correct and save the T1w image
        t2w_corrected = t2w_data / bias_data_dil
        self.save_output(t2w_corrected, t2w_img, "t2w_file", "T2w.nii.gz")
        t2w_corrected_brain = t2w_corrected * brain_mask
        self.save_output(t2w_corrected_brain, t2w_img,
                         "t2w_brain_file", "T2w_brain.nii.gz")

        np.seterr(**old_settings)

        return runtime

    def save_output(self, data, template, field, fname):

        full_fname = os.path.abspath(fname)
        self._results[field] = full_fname
        img = nib.Nifti1Image(data, template.affine, template.header)
        nib.save(img, full_fname)


correct_bias = Node(BiasCorrect(), "correct_bias")

# --- Build the workflow

workflow = Workflow(name="prisma_anat", base_dir="nipype_cache")

workflow.connect([
    (subject_source, file_input,
        [("subject", "subject")]),
    (file_input, reorient_t1w,
        [("t1w_files", "in_file")]),
    (file_input, reorient_t2w,
        [("t2w_files", "in_file")]),
    (reorient_t1w, register_t1w,
        [("out_file", "in_files")]),
    (reorient_t2w, register_t2w,
        [("out_file", "in_files")]),
    (register_t1w, normalize_linear,
        [("out_file", "in_file")]),
    (register_t1w, normalize_nonlinear,
        [("out_file", "in_file")]),
    (normalize_linear, normalize_nonlinear,
        [("out_matrix_file", "affine_file")]),
    (register_t1w, invert_warp,
        [("out_file", "reference")]),
    (normalize_nonlinear, invert_warp,
        [("fieldcoeff_file", "warp")]),
    (register_t1w, warp_mask,
        [("out_file", "ref_file")]),
    (invert_warp, warp_mask,
        [("inverse_warp", "field_file")]),
    (warp_mask, fill_mask,
        [("out_file", "in_file")]),
    #(register_t1w, register_between,
    #    [("out_file", "reference")]),
    #(register_t2w, register_between,
    #    [("out_file", "in_file")]),
    (register_t1w, register_between,
        [("out_file", "target_file")]),
    (register_t2w, register_between,
        [("out_file", "source_file")]),
    (register_t1w, correct_bias,
        [("out_file", "t1w_file")]),
    (register_between, correct_bias,
        [("registered_file", "t2w_file")]),
    #(register_between, correct_bias,
    #    [("out_file", "t2w_file")]),
    (fill_mask, correct_bias,
        [("out_file", "mask_file")]),
])


if __name__ == "__main__":

    workflow.write_graph("prisma_anat", "orig", "svg")
    workflow.config["crashdump_dir"] = os.path.realpath("crashdumps")
    workflow.run("MultiProc", dict(n_procs=8))
