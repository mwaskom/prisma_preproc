import os
import argparse
import nipype
from nipype import (Workflow, Node, MapNode, JoinNode,
                    Function, IdentityInterface, SelectFiles, DataSink)
from nipype.interfaces.base import (traits, BaseInterface,
                                    BaseInterfaceInputSpec, TraitedSpec,
                                    File, InputMultiPath, OutputMultiPath)
from nipype.interfaces import fsl, freesurfer as fs, utility

from lyman.tools.submission import submit_cmdline

data_dir = os.environ["SUBJECTS_DIR"]

# === Define pipeline nodes

# --- Parameter specification

"""

Due to concerns about motion changing the field we may want to have the
flexibility to use multiple fieldmaps per session. Then we could either have
the subject specify the mapping of fieldmaps to runs or compute all the
pairwise transformations and select the "closest" fieldmap for each run. Not a
bad idea!  But so much complexity!

"""

# --- Workflow parameterization

subject_source = Node(IdentityInterface(["subject"]),
                      name="subject_source",
                      iterables=("subject", ["rk"]))

session_source = Node(IdentityInterface(["subject", "session"]),
                      name="session_source",
                      itersource=("subject_source", "subject"),
                      iterables=("session", {"rk": [("rk", 1), ("rk", 2)]}))

run_source = Node(IdentityInterface(["subject", "session", "run"]),
                  name="run_source",
                  itersource=("session_source", "session"),
                  iterables=("run", {("rk", 1): [("rk", 1, 1),
                                                 ("rk", 1, 2)],
                                     ("rk", 2): [("rk", 2, 1),
                                                 ("rk", 2, 2),
                                                 ("rk", 2, 3)]}))


# --- Semantic information

def info_func(info_tuple):
    try:
        subject, session = info_tuple
        return subject, session
    except ValueError:
        subject, session, run = info_tuple
        return subject, session, run


sesswise_info = Node(Function("info_tuple",
                              ["subject", "session"],
                              info_func),
                     "sesswise_info")


runwise_info = Node(Function("info_tuple",
                             ["subject", "session", "run"],
                             info_func),
                    "runwise_info")

# --- Input file selection

session_templates = dict(
    se="{subject}/fieldmap/se_func_sess{session:02d}.nii.gz",
)

sesswise_input = Node(SelectFiles(session_templates,
                                  base_directory=data_dir),
                      "sesswise_input")


run_templates = dict(
    ts="{subject}/func/func_sess{session:02d}_run{run:02d}.nii.gz",
    sbref="{subject}/func/func_sess{session:02d}_run{run:02d}_sbref.nii.gz",
)

runwise_input = Node(SelectFiles(run_templates,
                                 base_directory=data_dir),
                     "runwise_input")


# --- Warpfield estimation using topup

# Distortion warpfield estimation
phase_encode_blips = ["y", "y", "y", "y-", "y-", "y-"]
readout_times = [1, 1, 1, 1, 1, 1]
estimate_distortions = Node(fsl.TOPUP(encoding_direction=phase_encode_blips,
                                      readout_times=readout_times,
                                      config="b02b0.cnf"),
                            "estimate_distortions")

# Average distortion-corrected spin-echo images
average_se = Node(fsl.MeanImage(out_file="se_restored.nii.gz"), "average_se")

# Select first warpfield image from output list
select_warp = Node(utility.Select(index=[0]), "select_warp")

# Define a mask of areas with large distortions
mask_distortions = Node(fsl.ImageMaths(op_string="-abs -thr 4 -Tmax -binv"),
                       "mask_distortions")

# --- Registration of SBRef to SE-EPI (with distortions)

sbref2se = Node(fsl.FLIRT(dof=6), "sbref2se")

# --- Registration of SE-EPI (without distortions) to Freesurfer anatomy

se2anat = Node(fs.BBRegister(init="fsl",
                             contrast_type="t2",
                             out_fsl_file="se2anat_flirt.mat",
                             out_reg_file="se2anat_tkreg.dat"),
               "se2anat")


# --- Definition of common cross-session space

class NativeTransformInput(BaseInterfaceInputSpec):

    session_info = traits.List(traits.Tuple())
    in_matrices = InputMultiPath(File(exists=True))
    in_volumes = InputMultiPath(File(exists=True))


class NativeTransformOutput(TraitedSpec):

    session_info = traits.List(traits.Tuple())
    out_matrices = OutputMultiPath(File(exists=True))
    out_flirt_file = File(exists=True)
    out_tkreg_file = File(exists=True)


class NativeTransform(BaseInterface):

    input_spec = NativeTransformInput
    output_spec = NativeTransformOutput

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["session_info"] = self.inputs.session_info

        out_matrices = [
            os.path.abspath("se2native_{:04d}.mat".format(i))
            for i, _ in enumerate(self.inputs.in_matrices, 1)
            ]

        outputs["out_matrices"] = out_matrices

        outputs["out_flirt_file"] = os.path.abspath("native2anat_flirt.mat")
        outputs["out_tkreg_file"] = os.path.abspath("native2anat_tkreg.dat")

        return outputs

    def _run_interface(self, runtime):

        subjects_dir = os.environ["SUBJECTS_DIR"]
        subj = set([s for s, _ in self.inputs.session_info]).pop()

        # Conver the anatomical image to nifit
        cmdline = ["mri_convert",
                   os.path.join(subjects_dir, subj, "mri/orig.mgz"),
                   "orig.nii.gz"]

        runtime = submit_cmdline(runtime, cmdline)

        # Compute the intermediate transform
        cmdline = ["midtrans",
                   "--template=orig.nii.gz",
                   "--separate=se2native_",
                   "--out=anat2native_flirt.mat"]
        cmdline.extend(self.inputs.in_matrices)

        runtime = submit_cmdline(runtime, cmdline)

        # Invert the anat2native transformation
        cmdline = ["convert_xfm",
                   "-omat", "native2anat_flirt.mat",
                   "-inverse",
                   "anat2native_flirt.mat"]

        runtime = submit_cmdline(runtime, cmdline)

        # Transform the first volume into the native space to get the geometry
        cmdline = ["flirt",
                   "-in", self.inputs.in_volumes[0],
                   "-ref", self.inputs.in_volumes[0],
                   "-init", "se2native_0001.mat",
                   "-out", "se_native_0001.nii.gz",
                   "-applyxfm"]

        runtime = submit_cmdline(runtime, cmdline)

        # Convert the FSL matrices to tkreg matrix format
        cmdline = ["tkregister2",
                   "--s", subj,
                   "--mov", "se_native_0001.nii.gz",
                   "--fsl", "native2anat_flirt.mat",
                   "--reg", "native2anat_tkreg.dat",
                   "--noedit"]

        runtime = submit_cmdline(runtime, cmdline)

        return runtime


se2native = JoinNode(NativeTransform(),
                     name="se2native",
                     joinsource="session_source",
                     joinfield=["session_info", "in_matrices", "in_volumes"])


# --- Associate native-space transformations with data from correct session

def select_transform_func(in_matrices, session_info, subject, session):

    for matrix, info in zip(in_matrices, session_info):
        if info == (subject, session):
            out_matrix = matrix
    return out_matrix


select_sesswise = Node(Function(["in_matrices", "session_info",
                                 "subject", "session"],
                                "out_matrix",
                                select_transform_func),
                           "select_sesswise")

select_runwise = select_sesswise.clone("select_runwise")

# --- Restore each sessions SE image in native space then average

split_se = Node(fsl.Split(dimension="t"), "split_se")

# TODO here (and below) we need to change the reference file to fix the affine
restore_se = MapNode(fsl.ApplyWarp(interp="spline", relwarp=True),
                     ["in_file", "ref_file", "premat", "field_file"],
                     "restore_se")

def flatten_file_list(in_files):
    out_files = [item for sublist in in_files for item in sublist]
    return out_files

combine_se = JoinNode(Function("in_files", "out_files", flatten_file_list),
                      name="combine_se",
                      joinsource="session_source",
                      joinfield=["in_files"])

merge_se = Node(fsl.Merge(dimension="t"), name="merge_se")

average_native = Node(fsl.MeanImage(out_file="se_native.nii.gz"),
                      name="average_native")

# --- Motion correction of timeseries to SBRef (with distortions)

ts2sbref = Node(fsl.MCFLIRT(save_mats=True), "ts2sbref")

# --- Combined motion correction and unwarping of timeseries

# Split the timeseries into each frame
split_ts = Node(fsl.Split(dimension="t"), "split_ts")

# Concatenation ts2sbref and sbref2se rigid transform
combine_rigids = MapNode(fsl.ConvertXFM(concat_xfm=True),
                         "in_file", "combine_rigids")

# Simultaneously apply rigid transform and nonlinear warpfield
restore_ts_frames = MapNode(fsl.ApplyWarp(interp="spline", relwarp=True),
                            ["in_file", "ref_file", "premat"], "restore_ts")

# Recombine the timeseries frames into a 4D image
merge_ts = Node(fsl.Merge(merged_file="ts_restored.nii.gz",
                          dimension="t"), "merge_ts")

# Save out important results
output_dir = os.path.realpath("python_script_outputs")
file_output = Node(DataSink(base_directory=output_dir),
                   "file_output")

# === Assemble pipeline

workflow = Workflow(name="prisma_preproc_multirun", base_dir="nipype_cache")

workflow.connect([
    (subject_source, session_source,
        [("subject", "subject")]),
    (subject_source, run_source,
        [("subject", "subject")]),
    (session_source, run_source,
        [("session", "session")]),
    (session_source, sesswise_info,
        [("session", "info_tuple")]),
    (run_source, runwise_info,
        [("run", "info_tuple")]),
    (sesswise_info, sesswise_input,
        [("subject", "subject"),
         ("session", "session")]),
    (runwise_info, runwise_input,
        [("subject", "subject"),
         ("session", "session"),
         ("run", "run")]),
    (sesswise_input, estimate_distortions,
        [("se", "in_file")]),
    (estimate_distortions, select_warp,
        [("out_warps", "inlist")]),
    (select_warp, mask_distortions,
        [("out", "in_file")]),
    (estimate_distortions, average_se,
        [("out_corrected", "in_file")]),
    (sesswise_info, se2anat,
        [("subject", "subject_id")]),
    (average_se, se2anat,
        [("out_file", "source_file")]),
    (session_source, se2native,
        [("session", "session_info")]),
    (sesswise_input, se2native,
        [("se", "in_volumes")]),
    (se2anat, se2native,
        [("out_fsl_file", "in_matrices")]),
    (se2native, select_sesswise,
        [("out_matrices", "in_matrices"),
         ("session_info", "session_info")]),
    (sesswise_info, select_sesswise,
        [("subject", "subject"),
         ("session", "session")]),
    (sesswise_input, split_se,
        [("se", "in_file")]),
    (split_se, restore_se,
        [("out_files", "in_file"),
         ("out_files", "ref_file")]),
    (estimate_distortions, restore_se,
        [("out_mats", "premat"),
         ("out_warps", "field_file")]),
    (select_sesswise, restore_se,
        [("out_matrix", "postmat")]),
    (restore_se, combine_se,
        [("out_file", "in_files")]),
    (combine_se, merge_se,
        [("out_files", "in_files")]),
    (merge_se, average_native,
        [("merged_file", "in_file")]),
    (runwise_input, ts2sbref,
        [("ts", "in_file"),
         ("sbref", "ref_file")]),
    (runwise_input, split_ts,
        [("ts", "in_file")]),
    (runwise_input, sbref2se,
        [("sbref", "in_file")]),
    (sesswise_input, sbref2se,
        [("se", "reference")]),
    (mask_distortions, sbref2se,
        [("out_file", "ref_weight")]),
    (ts2sbref, combine_rigids,
        [("mat_file", "in_file")]),
    (sbref2se, combine_rigids,
        [("out_matrix_file", "in_file2")]),
    (split_ts, restore_ts_frames,
        [("out_files", "in_file"),
         ("out_files", "ref_file")]),
    (combine_rigids, restore_ts_frames,
        [("out_file", "premat")]),
    (select_warp, restore_ts_frames,
        [("out", "field_file")]),
    (se2native, select_runwise,
        [("out_matrices", "in_matrices"),
         ("session_info", "session_info")]),
    (runwise_info, select_runwise,
        [("subject", "subject"),
         ("session", "session")]),
    (select_runwise, restore_ts_frames,
        [("out_matrix", "postmat")]),
    (restore_ts_frames, merge_ts,
        [("out_file", "in_files")]),
    (average_native, file_output,
        [("out_file", "@se_native")]),
    (merge_ts, file_output,
        [("merged_file", "@restored_timeseries")]),
    (se2native, file_output,
        [("out_tkreg_file", "@tkreg_file")]),
])

if __name__ == "__main__":

    workflow.write_graph("prisma_preproc_multirun", "exec", "svg")
    workflow.config["crashdump_dir"] = os.path.realpath("crashdumps")
    workflow.run("MultiProc", dict(n_procs=24))
