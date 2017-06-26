import os
import numpy as np
from scipy.ndimage import gaussian_filter
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
mni_mask = fsl.Info.standard_image("MNI152_T1_2mm_brain_mask.nii.gz")

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

dilate_mask = Node(fsl.DilateImage(operation="max"), "dilate_mask")

# --- Register the T2w to the T1w volume

robust_args = "--cost ROBENT --entradius 2 --entcorrection"
register_between = Node(fs.RobustRegister(est_int_scale=True,
                                          auto_sens=True,
                                          args=robust_args,
                                          registered_file="T2w.nii.gz"),
                        "register_between")

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
    bias_file = File(exists=True)
    smoothed_bias_file = File(exists=True)


class BiasCorrect(BaseInterface):

    input_spec = BiasCorrectInput
    output_spec = BiasCorrectOutput

    def _list_outputs(self):
        return self._results

    def run_interface(self, runtime):

        self._results = dict()

        # Load the input data
        t1w_img = nib.load(self.inputs.t1w_file)
        t2w_img = nib.load(self.inputs.t2w_file)
        t1w_data, t2w_data = t1w_img.get_data(), t2w_img.get_data()
        mask = nib.load(self.inputs.mask_file).get_data()

        # Compute and save the bias field
        bias_data = np.sqrt(np.abs(t1w_data * t2w_data))
        bias_data *= mask
        bias_data /= bias_data[mask.astype(bool)].mean()

        bias_img = nib.Nift1Image(bias_data, t1w_img.affine, t1w_img.header)
        bias_file = os.path.abspath("bias_field.nii.gz")
        self._results["bias_file"] = bias_file
        nib.save(bias_img, bias_file)

        # Define smoothing sigma in voxel units (note assumes isotropic)
        voxel_res = t1w_img.header.get_zooms()[0]
        voxel_sigma = self.inputs.smooth_sigma / voxel_res

        # Smooth the bias field within the brain mask
        smoothed_bias = gaussian_filter(bias_data, voxel_sigma)
        smoothed_mask = gaussian_filter(mask.astype(np.float), voxel_sigma)
        smoothed_bias /= smoothed_mask
        smoothed_bias *= mask
        smoothed_bias = np.nan_to_num(smoothed_bias)

        # Save out the smoothed bias field
        smoothed_bias_file = os.path.abspath("bias_field_smoothed.nii.gz")
        self._results["smoothed_bias_file"] = smoothed_bias_file
        nib.save(nib.Nifti1Image(smoothed_bias,
                                 t1w_img.affine, t1w_img.header),
                 smoothed_bias_file)

        # Bias-correct and save the T1w image
        t1w_corrected = np.nan_to_num(t1w_data / smoothed_bias)
        t1w_file = os.path.abspath("T1w.nii.gz")
        self._results["t1w_file"] = t1w_file
        nib.save(nib.Nifti1Image(t1w_corrected,
                                 t1w_img.affine, t1w_img.header), t1w_file)

        # Bias-correct and save the T1w image
        t2w_corrected = np.nan_to_num(t2w_data / smoothed_bias)
        t2w_file = os.path.abspath("T2w.nii.gz")
        self._results["t2w_file"] = t2w_file
        nib.save(nib.Nifti1Image(t2w_corrected,
                                 t2w_img.affine, t2w_img.header), t2w_file)

        return runtime


correct_bias = Node(BiasCorrect(), "compute_bias")

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
    (fill_mask, dilate_mask,
        [("out_file", "in_file")]),
    #(register_t1w, register_between,
    #    [("out_file", "reference")]),
    #(register_t2w, register_between,
    #    [("out_file", "in_file")]),
    #(dilate_mask, register_between,
    #    [("out_file", "ref_weight")]),
    (register_t1w, register_between,
        [("out_file", "target_file")]),
    (register_t2w, register_between,
        [("out_file", "source_file")]),
    (register_t1w, correct_bias,
        [("out_file", "t1w_file")]),
    (register_between, correct_bias,
        [("registered_file", "t2w_file")]),
    (dilate_mask, correct_bias,
        [("out_file", "mask_file")]),
])


if __name__ == "__main__":

    workflow.write_graph("prisma_anat", "orig", "svg")
    workflow.config["crashdump_dir"] = os.path.realpath("crashdumps")
    workflow.run("MultiProc", dict(n_procs=8))
