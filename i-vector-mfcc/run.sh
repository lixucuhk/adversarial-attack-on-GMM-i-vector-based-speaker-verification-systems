#!/bin/bash
# Copyright   2019   The Chinese University of Hong Kong (Author: Xu LI)
#

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

dataroot=/scratch/xli/Data_Source/Voxceleb
# The trials file is downloaded by local/make_voxceleb1_v2.pl.
voxceleb1_trials=data/voxceleb1_test/trials
voxceleb1_root=$dataroot/Voxceleb1
voxceleb2_root=$dataroot/Voxceleb2

stage=0

. ./parse_options.sh || exit 1;

if [ $stage -le 0 ]; then
  echo 'stage 0, prepare data-related files.'
  # This script creates data/voxceleb1_test and data/voxceleb1_train for latest version of VoxCeleb1.
  # Our evaluation set is the test portion of VoxCeleb1.
  local/make_voxceleb1_v2.pl $voxceleb1_root dev data/voxceleb1_train
  local/make_voxceleb1_v2.pl $voxceleb1_root test data/voxceleb1_test
  
  sed 's/.wav//g' $voxceleb1_trials >data/voxceleb1_test/trials_tmp || exit 1
  mv data/voxceleb1_test/trials_tmp $voxceleb1_trials
  # We'll train and test on Voxceleb1 only.
  # This should give 1211 speakers and 148642 utterances for the training portion.
  # If you want to involve other datasets for training, then combine them as the command below.
  # utils/combine_data.sh data/train data/voxceleb2_train data/voxceleb2_test data/voxceleb1_train
  cp -r data/voxceleb1_train data/train
fi

if [ $stage -le 1 ]; then
  echo 'stage 1, extract MFCCs features, and generate VAD files'
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in train voxceleb1_test; do
    steps/make_mfcc.sh --write-utt2num-frames true \
      --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

if [ $stage -le 2 ]; then
  echo 'stage 2, train GMM system.'
  # Train the UBM.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 4G" \
    --nj 20 --num-threads 8 --apply-cmn false \
    data/train 2048 \
    exp/diag_ubm

  sid/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
    --nj 20 --remove-low-count-gaussians false --apply-cmn false \
    data/train \
    exp/diag_ubm exp/full_ubm
fi

if [ $stage -le 3 ]; then
  echo 'stage 3, train i-vector system.'
  # In this stage, we train the i-vector extractor.
  #
  # Note that there are well over 1 million utterances in our training set,
  # and it takes an extremely long time to train the extractor on all of this.
  # Also, most of those utterances are very short.  Short utterances are
  # harmful for training the i-vector extractor.  Therefore, to reduce the
  # training time and improve performance, we will only train on the 100k
  # longest utterances.
  utils/subset_data_dir.sh \
    --utt-list <(sort -n -k 2 data/train/utt2num_frames | tail -n 100000) \
    data/train data/train_100k
  # Train the i-vector extractor.
  sid/train_ivector_extractor.sh --cmd "$train_cmd" \
    --ivector-dim 400 --num-iters 5 --apply-cmn false --nj 8 \
    exp/full_ubm/final.ubm data/train_100k \
    exp/extractor
fi

if [ $stage -le 4 ]; then
  echo 'stage 4, extract ivectors from both training and evaluation data.'
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 10 --apply-cmn false \
    exp/extractor data/train \
    exp/ivectors_train

  sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 10 --apply-cmn false \
    exp/extractor data/voxceleb1_test \
    exp/ivectors_voxceleb1_test
fi

if [ $stage -le 5 ]; then
  echo 'stage 5, train the PLDA model.'
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_train/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_train/ivector.scp \
    exp/ivectors_train/mean.vec || exit 1;

  # Train the PLDA model.
  $train_cmd exp/ivectors_train/log/plda.log \
    ivector-compute-plda ark:data/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train/ivector.scp ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/ivectors_train/plda || exit 1;
fi

if [ $stage -le 6 ]; then
  echo 'stage 6, plda scoring.'
  $train_cmd exp/scores/log/voxceleb1_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_train/plda - |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train/mean.vec scp:exp/ivectors_voxceleb1_test/ivector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train/mean.vec scp:exp/ivectors_voxceleb1_test/ivector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_test || exit 1;
fi

if [ $stage -le 7 ]; then
  echo 'stage 7, EER performance.'
  eer=`compute-eer <(python local/prepare_for_eer.py $voxceleb1_trials exp/scores_voxceleb1_test) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"

fi
