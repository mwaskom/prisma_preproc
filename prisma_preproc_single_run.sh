#! /bin/bash

# Beta version of a script to preprocess SMS-EPI data from the Prisma scanner
# Written by Michael Waskom | Last updated June 15, 2017

if [ "$#" -ne 6 ]; then
    echo '
USAGE: prisma_preproc.sh <subj> <ts> <sbref> <se> <se_rev> <wd>

Parameter Information:

  subj: Freesurfer subject ID
  ts : 4D EPI timeseries image
  sbref : 3D single-band EPI reference image for the timeseries
  se : 4D spin echo EPI image with phase encoding corresponding to timeseries
  se_rev : 4D spin echo EPI image with reversed phase encoding
  wd : working directory for writing intermediate outputs

Notes:

- Assumes that the anatomical data for subject <subj> have been processed with
  recon-all and that $SUBJECTS_DIR is set correctly
- Assumes that phase encoding was performed in either the AP or PA direction
- Does not currently write files anywhere other than <wd>

'
    exit 1
fi

# Don't continue past errors
set -e

# Process command line arguments
subj=$1
ts=$2
sbref=$3
se=$4
se_rev=$5
wd=$6

# Ensure working directory is exists
mkdir -p $wd

# --- Warpfield estimation using TOPUP

# Create a phase encoding parameters file to use
se_params=$wd/se_params.txt

if [ -e $se_params ]; then
    rm $se_params
fi

se_frames=`fslval $se dim4`

for i in $(seq 1 $se_frames); do
    echo "0 1 0 1" >> $se_params
done
for i in $(seq 1 $se_frames); do
    echo "0 -1 0 1" >> $se_params
done

# Concatenate the pair of spin-echo scans
se_pair=$wd/se_pair.nii.gz
fslmerge -t $se_pair $se $se_rev

# Run topup to estimate the distortions
topup --imain=$se_pair \
      --datain=$se_params \
      --config=b02b0.cnf \
      --out=$wd/topup \
      --iout=$wd/se_pair_unwarped \
      --dfout=$wd/topup_warp \
      --rbmout=$wd/topup_xfm \
      --verbose

# Average the unwarped spin echo images
se_unwarped=$wd/se_unwarped.nii.gz
fslmaths $wd/se_pair_unwarped -Tmean $wd/se_unwarped.nii.gz

warpfield=$wd/topup_warp_01.nii.gz

# --- Registration of SBRef to SE space (with distortions)

sbref2se=$wd/xfm_sbref2se.mat

flirt -in $sbref \
      -ref $se \
      -omat $sbref2se \
      -dof 6 \
      -v

# --- Distortion correction (unwarping) of SBRef image

sbref_unwarped=$wd/sbref_unwarped.nii.gz

applywarp -i $sbref \
          -r $sbref \
          -o $sbref_unwarped \
          -w $warpfield \
          --premat=$sbref2se \
          --interp=spline \
          --rel \
          --verbose

# --- Registration of the spin echo EPI (without distortions) to the anatomy

se2anat_tkreg=$wd/xfm_se2anat.dat
se2anat_flirt=$wd/xfm_se2anat.mat

bbregister --s $subj \
           --t2 \
           --mov $se_unwarped \
           --reg $se2anat_tkreg \
           --fslmat $se2anat_flirt

# --- Motion correction of the timeseries to the SBRef (with distortions)

mcflirt -in $ts \
        -refvol $sbref \
        -out $wd/ts_mc \
        -mats \
        -v

ts2sbref_dir=$wd/xfm_ts2sbref.d

mv $wd/ts_mc.mat $wd/xfm_ts2sbref.d
rm $wd/ts_mc.nii.gz

# --- Combined motion correction and unwarping of the timeseries

ts2se_dir=$wd/xfm_ts2se.d/
ts_frame_dir=$wd/ts.d/
ts_mc_unwarped_frame_dir=$wd/ts_mc_unwarped.d/

ts_frames=`fslval $ts dim4`

mkdir -p $ts2se_dir
mkdir -p $ts_frame_dir
mkdir -p $ts_mc_unwarped_frame_dir

# Split the timeseries into frames
fslsplit $ts $ts_frame_dir -t

ts_mc_unwarped_frame_names=""

for i in $(seq -w 0 `expr $ts_frames - 1`); do

    echo "Motion correcting/unwarping frame $i"

    if [ ${#i} -eq 3 ]; then
        i="0$i"
    fi

    # Combine the TS > SBRef and SBRef > SE rigid transforms
    convert_xfm -omat $ts2se_dir/$i.mat \
                -concat \
                $sbref2se \
                $ts2sbref_dir/MAT_$i

    # Apply the combined rigid and nonlinear unwarping
    unwarped_frame=$ts_mc_unwarped_frame_dir/$i.nii.gz
    applywarp -i $ts_frame_dir/$i.nii.gz \
              -r $ts_frame_dir/$i.nii.gz \
              -w $warpfield \
              -o $unwarped_frame \
              --premat=$ts2se_dir/$i.mat \
              --interp=spline \
              --rel

    ts_mc_unwarped_frame_names="$ts_mc_unwarped_frame_names $unwarped_frame"

done

# Concatenate the timeseries frames
ts_mc_unwarped=$wd/ts_mc_unwarped.nii.gz
fslmerge -t $ts_mc_unwarped $ts_mc_unwarped_frame_names

# --- Transformation of the white and pial surfaces into functional space

for surf in white pial; do
    for hemi in lh rh; do

        mri_surf2surf --s $subj \
                      --hemi $hemi \
                      --sval-xyz $surf \
                      --tval $wd/$hemi.$surf \
                      --tval-xyz $sbref_unwarped \
                      --reg $se2anat_tkreg

    done
done
