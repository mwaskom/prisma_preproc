import os
from nipype import (Workflow, Node, MapNode,
                    Function, IdentityInterface, SelectFiles, DataSink)
from nipype.interfaces import fsl, freesurfer as fs, utility


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
                                  ref_weight=mni_mask,
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

fill_mask = Node(fsl.ImageMaths(op_string="fillh"), "fill_mask")

dilate_mask = Node(fsl.DilateImage(operation="max"), "dilate_mask")

# --- Register the T2w to the T1w volume

# Technically better but takes forever, and given smoothing of
# bias field the difference is likely of no consequence
# robust_args = "--cost ROBENT --entradius 2 --entcorrection"
# register_between = Node(fs.RobustRegister(est_int_scale=True,
#                                           auto_sens=True,
#                                           args=robust_args,
#                                           registered_file="T2w.nii.gz"),
#                         "register_between")

register_between = Node(fsl.FLIRT(dof=6, interp="spline"),
                        "register_between")

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
    (register_t1w, register_between,
        [("out_file", "reference")]),
    (register_t2w, register_between,
        [("out_file", "in_file")]),
    (dilate_mask, register_between,
        [("out_file", "ref_weight")]),
])


if __name__ == "__main__":

    workflow.write_graph("prisma_anat", "orig", "svg")
    workflow.config["crashdump_dir"] = os.path.realpath("crashdumps")
    workflow.run("MultiProc", dict(n_procs=8))
