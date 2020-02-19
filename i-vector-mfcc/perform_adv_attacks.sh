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

stage=0

. ./parse_options.sh || exit 1

voxceleb1_root=$dataroot/Voxceleb1
voxceleb2_root=$dataroot/Voxceleb2

## In this script, we rebuild the forward process in SV systems by Pytorch codes, 
## and the model parameters (GMM, i-vector extractor and PLDA) used are those 
## trained in the `run.sh' code. To access the model parameters, we first convert
## kaldi-format parameters into txt format using kaldi commands. Especially, the 
## `copy-ivectorextractor' command is self-developed to convert i-vector extractor
## parameters into the `txt' format. You need to copy it from ./copy-ivectorextractor.cc
## to /your_path/kaldi/src/ivectorbin/copy-ivectorextractor.cc first, then compile it for usage.
## This cc file is the same as the one in the i-vector-lpms directory, so you just need do only once.

if [ $stage -le 0 ]; then
	echo 'stage 0, convert kaldi-format files into txt format.'
	fgmm-global-copy --binary=false exp/full_ubm/final.ubm exp/full_ubm/final_ubm.txt || exit 1
	copy-ivectorextractor --binary=false exp/extractor/final.ie exp/extractor/final_ie.txt || exit 1
	ivector-copy-plda --binary=false exp/ivectors_train/plda exp/ivectors_train/plda.txt || exit 1

	mkdir -p data/voxceleb1_test/voiced_feats
	mkdir -p data/voxceleb1_test/trials_pieces
	mkdir -p data/voxceleb1_test/gradients

	select-voiced-frames scp:data/voxceleb1_test/feats.scp scp:data/voxceleb1_test/vad.scp \
	                     ark,scp:data/voxceleb1_test/voiced_feats/voiced_feats.ark,data/voxceleb1_test/voiced_feats/voiced_feats.scp || exit 1
	
	cp data/voxceleb1_test/trials data/voxceleb1_test/trials_pieces/trials
	cp exp/ivectors_voxceleb1_test/ivector.scp exp/ivectors_voxceleb1_test/enrollivector.scp 

	# generate adversarial trials for later usage.
	python3 gen_adv_trials.py --ori-trials data/voxceleb1_test/trials --adv-trials data/voxceleb1_test/trials_adv || exit 1
fi

if [ $stage -le 1 ]; then
	echo 'stage 1, compute adversarial gradients using FGSM.'

	python3 adv_grad_compute.py --fgmmfile exp/full_ubm/final_ubm.txt --datafile data/voxceleb1_test/voiced_feats/voiced_feats.scp \
			        --ivector-extractor exp/extractor/final_ie.txt --plda-mdl exp/ivectors_train/plda.txt \
			        --enroll-ivectors exp/ivectors_voxceleb1_test/enrollivector.scp --trials data/voxceleb1_test/trials \
			        --ivector-global-mean exp/ivectors_train/mean.vec --gradfile-ark data/voxceleb1_test/gradients/signed_gradientfile.ark \
			        --gradfile-scp data/voxceleb1_test/gradients/signed_gradientfile.scp || exit 1

fi

if [ $stage -le 2 ]; then
	echo 'stage 2, generate adversarial voiced feats from gradients.'
	# For MFCC-ivec system, we only generate adversarial samples at acoustic feature level (voiced feats), and not at wav file level due to the 
	# irreversibility from wav to MFCC features. These adversarial samples are used to perform white box attack on MFCC-ivec system.
	

	# $sigma is a parameter of adjusting perturbation degree
	sigma=1.0

	echo 'generate samples at acoustic feature level.'
	python3 local/gen_adv_voiced_feats.py --grads data/voxceleb1_test/gradients/signed_gradientfile.scp \
			--ori-feats data/voxceleb1_test/feats.scp --vad data/voxceleb1_test/vad.scp --sigma $sigma || exit 1
	
fi

if [ $stage -le 3 ]; then
	echo 'stage 3, generate adversarial acoustic feats from gradients. '
	# Different from the stage 2, the previously removed nonspeech frames are added \
	# back to the adversarial voiced feats to generate the adversarial acousic feats. \
	# These feats are used to perform black box attack on the xvec-MFCC system (the cross-model architecture setting).

	# $sigma is a parameter of adjusting perturbation degree
	sigma=1.0

	python3 local/gen_adv_feats.py --grads data/voxceleb1_test/gradients/signed_gradientfile.scp \
			--ori-feats data/voxceleb1_test/feats.scp --vad data/voxceleb1_test/vad.scp --sigma $sigma || exit 1

	python3 local/gen_adv_vad.py --ori-vad data/voxceleb1_test/vad.scp \
	                             --adv-feats data/voxceleb1_test/spoofed_feats_sigma${sigma}.scp \
	                             --adv-vad data/voxceleb1_test/spoofed_vad.scp || exit 1
	
fi

if [ $stage -le 4 ]; then
	echo 'stage 4, using adversarial samples for attacking MFCC-ivec system. (the white box attack)'

	# $sigma is a parameter of perturbation degree
	sigma=1.0
	python3 speaker_verification.py --sigma $sigma --fgmmfile exp/full_ubm/final_ubm.txt \
				--datafile data/voxceleb1_test/spoofed_voiced_feats_sigma${sigma}.scp \
				--ivector-extractor exp/extractor/final_ie.txt --plda-mdl exp/ivectors_train/plda.txt \
				--enroll-ivectors exp/ivectors_voxceleb1_test/enrollivector.scp --trials data/voxceleb1_test/trials_adv \
				--ivector-global-mean exp/ivectors_train/mean.vec || exit 1

	# compute EER
	python3 local/evaluation_metric.py --trials data/voxceleb1_test/trials_adv \
									--scores exp/verification_scores_sigma${sigma} || exit 1

	echo 'adjust operation point below, and then continue.'
	exit 0
	# given operation point, compute FAR and FRR. The operation point could be shown in the 
	# output from the last command. And you set `operation_point' at that value.
	operation_point=-3.5847
	python3 local/evaluation_metric.py --trials data/voxceleb1_test/trials_adv \
									--scores exp/verification_scores_sigma${sigma} --threshold ${operation_point} || exit 1

fi

if [ $stage -le 5 ]; then
	echo 'stage 5, using adversarial samples from lpms-ivec system for attacking MFCC-ivec system. (the black box attack of the cross-feature setting)'

	# $sigma is a parameter of perturbation degree
	sigma=1.0

	# extract acoustic features from adversarial wav files
	./pre_spoofed_data.sh $sigma || exit 1

	# compute verification scores under adversarial attack
	python3 speaker_verification_adv_audios.py --sigma $sigma --fgmmfile exp/full_ubm/final_ubm.txt \
				--datafile data/lpms_ivec_adv_sigma$sigma/voiced_feats.scp \
				--ivector-extractor exp/extractor/final_ie.txt --plda-mdl exp/ivectors_train/plda.txt \
				--enroll-ivectors exp/ivectors_voxceleb1_test/enrollivector.scp --trials data/voxceleb1_test/trials_adv \
				--ivector-global-mean exp/ivectors_train/mean.vec || exit 1
				
	# compute EER (13.84%)
	python3 local/evaluation_metric.py --trials data/voxceleb1_test/trials_adv \
									--scores exp/lpms_ivec_adv_verification_scores_sigma${sigma} || exit 1

	# given operation point, compute FAR (17.03%) and FRR (11.15%).
	operation_point=-3.5847
	python3 local/evaluation_metric.py --trials data/voxceleb1_test/trials_adv \
									--scores exp/lpms_ivec_adv_verification_scores_sigma${sigma} --threshold ${operation_point} || exit 1
fi
