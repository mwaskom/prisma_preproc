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


class NativeTransformOutput(TraitedSpec):

    session_info = traits.List(traits.Tuple())
    out_matrices = OutputMultiPath(File(exists=True))


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

        return outputs

    def _run_interface(self, runtime):

        cmdline = ["midtrans",
                   "--separate=se2native_",
                   "--out=anat2native.mat"]
        cmdline.extend(self.inputs.in_matrices)

        runtime = submit_cmdline(runtime, cmdline)

        return runtime


se2native = JoinNode(NativeTransform(),
                     name="se2native",
                     joinsource="session_source",
                     joinfield=["session_info", "in_matrices"])


def select_transform_func(in_matrices, session_info, subject, session):

    for matrix, info in zip(in_matrices, session_info):
        if info == (subject, session):
            out_matrix = matrix
    return out_matrix


select_transform = Node(Function(["in_matrices", "session_info",
                                  "subject", "session"],
                                 "out_matrix",
                                 select_transform_func),
                        "select_transform")

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
    (estimate_distortions, average_se,
        [("out_corrected", "in_file")]),
    (sesswise_info, se2anat,
        [("subject", "subject_id")]),
    (average_se, se2anat,
        [("out_file", "source_file")]),
    (session_source, se2native,
        [("session", "session_info")]),
    (se2anat, se2native,
        [("out_fsl_file", "in_matrices")]),
    (se2native, select_transform,
        [("out_matrices", "in_matrices"),
         ("session_info", "session_info")]),
    (runwise_info, select_transform,
        [("subject", "subject"),
         ("session", "session")]),
    (runwise_input, ts2sbref,
        [("ts", "in_file"),
         ("sbref", "ref_file")]),
    (runwise_input, split_ts,
        [("ts", "in_file")]),
    (runwise_input, sbref2se,
        [("sbref", "in_file")]),
    (sesswise_input, sbref2se,
        [("se", "reference")]),
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
    (select_transform, restore_ts_frames,
        [("out_matrix", "postmat")]),
    (restore_ts_frames, merge_ts,
        [("out_file", "in_files")]),
    (average_se, file_output,
        [("out_file", "@average_se")]),
    (merge_ts, file_output,
        [("merged_file", "@restored_timeseries")]),
    (se2anat, file_output,
        [("out_reg_file", "@tkreg_file")]),
])

if __name__ == "__main__":

    workflow.write_graph("prisma_preproc_multirun", "exec", "svg")
    workflow.config["crashdump_dir"] = os.path.realpath("crashdumps")
    workflow.run("MultiProc", dict(n_procs=24))
