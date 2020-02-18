#!/bin/bash
# Copyright 2015-2017   David Snyder
#                2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2015   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e

sigma=$1
wav_dir=../i-vector-lpms/adv_audios/sigma$sigma

data_dir=data/lpms_ivec_adv_sigma$sigma
mfccdir=mfcc/lpms_ivec_adv_sigma$sigma
vaddir=mfcc/lpms_ivec_adv_sigma$sigma

# echo "Extracting wav file..."
# mkdir -p $data_dir
# [ -f $data_dir/wav.scp ] && rm -f $data_dir/wav.scp
# [ -f $data_dir/spk2utt ] && rm -f $data_dir/spk2utt
# [ -f $data_dir/utt2spk ] && rm -f $data_dir/utt2spk

# for file in $wav_dir/*; do
# 	uttid=$(basename $file .wav)
# 	if [ $((`echo $uttid | awk '{print length($0)}'`)) -gt 30 ]; then
# 		echo $uttid;
# 		echo "${uttid} $file" >>$data_dir/wav.scp;
# 		echo "${uttid} ${uttid}" >>$data_dir/utt2spk;
# 		echo "${uttid} ${uttid}" >>$data_dir/spk2utt;
# 	fi
# done

# echo "Extracting MFCCs..."
# steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 60 --cmd "$train_cmd" \
#   $data_dir exp/make_mfcc $mfccdir

# echo "Computing VAD..."
# sid/compute_vad_decision.sh --nj 60 --cmd "$train_cmd" \
#   $data_dir exp/make_vad $vaddir

# utils/fix_data_dir.sh $data_dir

select-voiced-frames scp:${data_dir}/feats.scp scp:${data_dir}/vad.scp ark,scp:${data_dir}/voiced_feats.ark,${data_dir}/voiced_feats.scp

# cat data/voxceleb1_test/voiced_feats.scp >>${data_dir}/voiced_feats.scp
wc -l ${data_dir}/voiced_feats.scp
