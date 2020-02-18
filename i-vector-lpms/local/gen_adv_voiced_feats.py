import sys, os
import argparse
import numpy as np 
import subprocess
from tqdm import tqdm
from preprocess.generic import load_wav_snf, stft, save_wav, \
     revert_power_db_to_wav, uttid2wav, load_wav, save_wav_snf, extract_adv_voiced_feats

import kaldi_io

def main(args):
	keys, mat_list = extract_adv_voiced_feats(args.grads, args.vad, args.ori_feats, args.sigma)
	adv_featsfile = 'ark:| copy-feats ark: ark,scp:data/voxceleb1_test/spoofed_voiced_feats_sigma'+str(args.sigma)+'.ark,data/voxceleb1_test/spoofed_voiced_feats_sigma'+str(args.sigma)+'.scp'
	utts_done = 0

	with kaldi_io.open_or_fd(adv_featsfile, 'wb') as f:
		for key, mat in zip(keys, mat_list):
			kaldi_io.write_mat(f, mat, key=key)
		utts_done += 1
		print('%d done.' %(utts_done))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Attack on GMM ivector+PLDA SV systems.')
    parser.add_argument('--grads', 
        default="data/voxceleb1_test/gradientfile.scp", type=str)
    parser.add_argument('--ori-feats', default="data/voxceleb1_test/feats.scp", type=str)
    parser.add_argument('--vad', default="data/voxceleb1_test/vad.scp", type=str)
    # parser.add_argument('--testaudios', default="data/voxceleb1_test/wav.scp", type=str)
    # parser.add_argument('--out_dir', default="adv_audios/sigma", type=str)
    parser.add_argument('--sigma', default=5, type=float)
    # parser.add_argument('--mdl_save_dir', default=None, type=str,
    #                     help='Path to the model save directory(default: None)')
    # parser.add_argument('-d', '--device', default=None, type=str,
    #                     help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()
    main(args)
