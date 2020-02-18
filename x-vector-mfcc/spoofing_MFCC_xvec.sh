#!/bin/bash
# Copyright   2019   The Chinese University of Hong Kong (Author: Xu LI)
#

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc


dataroot=/scratch/xli/Data_Source/Voxceleb
# The trials file is copied from i-vector-lpms system.
voxceleb1_trials=data/voxceleb1_test/trials_adv

# $sigma is a parameter of adjusting perturbation degree
sigma=1.0
use_gpu=false
nj=5
stage=0

. ./parse_options.sh || exit 1

voxceleb1_root=$dataroot/Voxceleb1

# copy adversarial trials file
cp ../i-vector-lpms/data/voxceleb1_test/trials_adv data/voxceleb1_test/

if [ $stage -le 0 ]; then
  echo 'stage 0, spoof MFCC-xvec by adversarial samples from LPMS-ivec system. (cross-feature-model attack)'

  ./spoofing_sv_lpms.sh $sigma $voxceleb1_trials $use_gpu $nj || exit 1

  # compute EER
  python3 local/evaluation_metric.py --trials data/voxceleb1_test/trials_adv \
								--scores exp/scores_lpms_ivec_adv_sigma${sigma} || exit 1

  # given original operation point (obtained by the command below), compute new FAR (9.41%) and FRR (9.35%).
  # python3 local/evaluation_metric.py --trials data/voxceleb1_test/trials \
  #                 --scores exp/scores_voxceleb1_test || exit 1

  operation_point=-3.774
  python3 local/evaluation_metric.py --trials data/voxceleb1_test/trials_adv \
								--scores exp/scores_lpms_ivec_adv_sigma${sigma} --threshold ${operation_point} || exit 1
  # exit 0
fi

if [ $stage -le 1 ]; then
  echo 'stage 1, spoof MFCC-xvec by adversarial samples from MFCC-ivec system. (cross-model architecture attack)'
  ./spoofing_sv_mfcc.sh $sigma $voxceleb1_trials $use_gpu $nj || exit 1
  
  # compute EER
  python3 local/evaluation_metric.py --trials data/voxceleb1_test/trials_adv \
								--scores exp/scores_mfcc_ivec_adv_sigma${sigma} || exit 1

  # given original operation point, compute FAR (13.79%) and FRR (14.28%).
  operation_point=-3.774
  python3 local/evaluation_metric.py --trials data/voxceleb1_test/trials_adv \
								--scores exp/scores_mfcc_ivec_adv_sigma${sigma} --threshold ${operation_point} || exit 1
fi
