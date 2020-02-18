#!/bin/bash
# Copyright   2019   The Chinese University of Hong Kong (Author: Xu LI)
#

. ./cmd.sh
. ./path.sh
set -e

sigma=$1
voxceleb1_trials=$2 # trials / trials_adv
use_gpu=$3
nj=$4
stage=0

nnet_dir=exp/xvector_nnet_1a
data_dir=data/lpms_ivec_adv_sigma$sigma
mfccdir=mfcc/lpms_ivec_adv_sigma$sigma
vaddir=mfcc/lpms_ivec_adv_sigma$sigma

if [ $stage -le 0 ]; then
  echo "Copying input data from MFCC-ivec..."
  mkdir -p $data_dir
  cp -r ../i-vector-mfcc/${data_dir}/{feats.scp,frame_shift,utt2num_frames,vad.scp,wav.scp,utt2spk} $data_dir/

  utils/fix_data_dir.sh $data_dir
fi

if [ $stage -le 1 ]; then
  echo "Extractin Xvectors..."
  # Extract x-vectors for centering, LDA, and PLDA training.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj $nj --use-gpu $use_gpu \
    $nnet_dir $data_dir \
    $nnet_dir/xvectors_lpms_ivec_adv_sigma${sigma}

fi

if [ $stage -le 2 ]; then
  echo "Combining original and spoofed xvector scp..."
  cat $nnet_dir/xvectors_voxceleb1_test/xvector.scp >>$nnet_dir/xvectors_lpms_ivec_adv_sigma${sigma}/xvector.scp
  wc -l $nnet_dir/xvectors_lpms_ivec_adv_sigma${sigma}/xvector.scp
fi

if [ $stage -le 11 ]; then
  $train_cmd exp/scores/log/xvectors_lpms_ivec_adv_sigma${sigma}_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train/plda - |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_lpms_ivec_adv_sigma${sigma}/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_lpms_ivec_adv_sigma${sigma}/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_lpms_ivec_adv_sigma${sigma} || exit 1;
fi
