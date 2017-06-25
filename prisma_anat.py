import os
import nipype
from nipype import (Workflow, Node, MapNode,
                    Function, IdentityInterface, SelectFiles, DataSink)
from nipype.interfaces import fsl, freesurfer as fs, utility


data_dir = os.environ["SUBJECTS_DIR"]

subject_source = Node(IdentityInterface(["subject"]),
                      name="subject_source",
                      iterables=("subject", ["rk"]))


templates = dict(t1w_files="{subject}/anat/T1w_*.nii.gz",
                 t2w_files="{subject}/anat/T2w_*.nii.gz")
file_input = Node(SelectFiles(templates,
                              base_directory=data_dir),
                  "file_input")


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

robust_args = "--cost ROBENT --entradius 2 --entcorrection"
register_between = Node(fs.RobustRegister(est_int_scale=True,
                                          auto_sens=True,
                                          args=robust_args,
                                          registered_file="T2w.nii.gz"),
                        "register_between")

workflow = Workflow(name="prisma_anat", base_dir="nipype_cache")

workflow.connect([
    (subject_source, file_input,
        [("subject", "subject")]),
    (file_input, register_t1w,
        [("t1w_files", "in_files")]),
    (file_input, register_t2w,
        [("t2w_files", "in_files")]),
    (register_t1w, register_between,
        [("out_file", "target_file")]),
    (register_t2w, register_between,
        [("out_file", "source_file")]),
])


if __name__ == "__main__":

    workflow.write_graph("prisma_anat", "orig", "svg")
    workflow.config["crashdump_dir"] = os.path.realpath("crashdumps")
    workflow.run("MultiProc", dict(n_procs=8))
