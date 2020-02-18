import sys, os
import argparse
import numpy as np 
import subprocess
from tqdm import tqdm
from preprocess.generic import load_wav_snf, stft, save_wav, \
     revert_power_db_to_wav, extract_adv_mat, uttid2wav, load_wav, save_wav_snf, extract_adv_mat_frm_grads

NUM_UTTS = 5
NUM_FRAMES = 600




def main(args):
	# keys, mat_list = extract_adv_mat(args.spoofed_feats, args.vad, args.ori_feats)
	# print(1+args.sigma)
	keys, mat_list = extract_adv_mat_frm_grads(args.grads, args.vad, args.ori_feats, args.sigma)
	outdir = args.out_dir+str(args.sigma)
	os.makedirs(outdir, exist_ok=True)
	uttid2wav_dict = uttid2wav(args.testaudios)

	for key, mat in zip(keys, mat_list):
		testkey = key[26:]
		ori_audio = uttid2wav_dict.get(testkey)
		wav = load_wav_snf(ori_audio)
		print('computing original stft.')
		spec = stft(wav, n_fft=128, hop_length=64, win_length=128, window="blackman")
		print('reverting.')
		adv_wav = revert_power_db_to_wav(spec, mat, n_fft=128, hop_length=64, win_length=128, window="blackman")
		print('saving.')
		save_wav_snf(wav, outdir+'/'+testkey+'.wav')
		save_wav_snf(adv_wav, outdir+'/'+key+'.wav')
		print('%s done.' %(key))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Attack on GMM ivector+PLDA SV systems.')
    parser.add_argument('--grads', 
        default="data/voxceleb1_test/gradientfile.scp", type=str)
    parser.add_argument('--ori_feats', default="data/voxceleb1_test/feats.scp", type=str)
    parser.add_argument('--vad', default="data/voxceleb1_test/vad.scp", type=str)
    parser.add_argument('--testaudios', default="data/voxceleb1_test/wav.scp", type=str)
    parser.add_argument('--out_dir', default="adv_audios/sigma", type=str)
    parser.add_argument('--sigma', default=5, type=float)
    # parser.add_argument('--mdl_save_dir', default=None, type=str,
    #                     help='Path to the model save directory(default: None)')
    # parser.add_argument('-d', '--device', default=None, type=str,
    #                     help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()
    main(args)
