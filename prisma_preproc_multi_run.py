import os
import argparse
import nipype
from nipype import (Workflow, Node, MapNode,
                    IdentityInterface, SelectFiles, DataSink)
from nipype.interfaces import fsl, freesurfer as fs, utility

data_dir = os.environ["SUBJECTS_DIR"]

# === Define pipeline nodes

# --- Parameter specification

# TODO there are a number of options for how to implement the parameterization
# of the workflow. They key is that we need two levels of iterables: one on the
# (subject, session) pairs and one on the runs. This is not neccessarily the
# optimal implementation of how to do that.

# Separately, we need to figure out how we want to ask the user to encode the
# subject/session/run information.

session_source = Node(IdentityInterface(["session"]),
                      name="session_source",
                      iterables=("session", [("rk", 1), ("rk", 2)]))


def session_info_func(session):
    subject, session = session
    return subject, session 
session_info = Node(utility.Function("session", ["subject", "session"],
                                     session_info_func),
                    "session_info")


run_source = Node(IdentityInterface(["session", "run"]),
                  name="run_source",
                  itersource=("session_source", "session"),
                  iterables=("run", {("rk", 1): [1, 2],
                                     ("rk", 2): [1, 2, 3]}))

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

se2anat = Node(fs.BBRegister(contrast_type="t2",
                             out_reg_file="se2anat_tkreg.dat"),
               "se2anat")

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
    (session_source, session_info,
        [("session", "session")]),
    (session_source, run_source,
        [("session", "session")]),
    (session_info, sesswise_input,
        [("subject", "subject"),
         ("session", "session")]),
    (session_info, runwise_input,
        [("subject", "subject"),
         ("session", "session")]),
    (run_source, runwise_input,
        [("run", "run")]),
    (sesswise_input, estimate_distortions,
        [("se", "in_file")]),
    (estimate_distortions, select_warp,
        [("out_warps", "inlist")]),
    (estimate_distortions, average_se,
        [("out_corrected", "in_file")]),
    (session_info, se2anat,
        [("subject", "subject_id")]),
    (average_se, se2anat,
        [("out_file", "source_file")]),
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
        [("out_files", "in_file"), ("out_files", "ref_file")]),
    (combine_rigids, restore_ts_frames,
        [("out_file", "premat")]),
    (select_warp, restore_ts_frames,
        [("out", "field_file")]),
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

    workflow.write_graph("prisma_preproc", "orig", "svg")
    workflow.config["crashdump_dir"] = os.path.realpath("crashdumps")
    workflow.run("MultiProc", dict(n_procs=24))
