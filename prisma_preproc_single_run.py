import os
import argparse
import nipype
from nipype import Workflow, Node, MapNode, SelectFiles, DataSink
from nipype.interfaces import fsl, freesurfer as fs, utility


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-subj", help="Freesurfer subject ID")
parser.add_argument("-ts", help="4D EPI timeseries image")
parser.add_argument("-sbref", help="3D single-band EPI reference image")
parser.add_argument("-se", help="Matched phase encode spin echo image")
parser.add_argument("-serev", help="Reversed phase encode spin echo image")
args = parser.parse_args()

# === Define pipeline nodes

# --- Input file selection

files = dict(ts=args.ts,
             sbref=args.sbref,
             se=args.se,
             serev=args.serev)

current_dir = os.path.realpath(".")
file_input = Node(SelectFiles(files, base_directory=current_dir), "file_input")

# --- Warpfield estimation using topup

list_se = Node(utility.Merge(2), "list_se")

# Combine different spin echo images into one file
merge_se = Node(fsl.Merge(dimension="t"), "merge_se")

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

se2anat = Node(fs.BBRegister(subject_id=args.subj,
                             contrast_type="t2",
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

workflow = Workflow(name="prisma_preproc", base_dir="nipype_cache")

workflow.connect([
    (file_input, list_se,
        [("se", "in1"), ("serev", "in2")]),
    (list_se, merge_se,
        [("out", "in_files")]),
    (merge_se, estimate_distortions,
        [("merged_file", "in_file")]),
    (estimate_distortions, select_warp,
        [("out_warps", "inlist")]),
    (estimate_distortions, average_se,
        [("out_corrected", "in_file")]),
    (average_se, se2anat,
        [("out_file", "source_file")]),
    (file_input, ts2sbref,
        [("ts", "in_file"), ("sbref", "ref_file")]),
    (file_input, split_ts,
        [("ts", "in_file")]),
    (file_input, sbref2se,
        [("sbref", "in_file"), ("se", "reference")]),
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
