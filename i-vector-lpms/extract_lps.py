import json
import codecs
import os, sys
import numpy as np 
import argparse
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from tqdm import tqdm
from local.preprocess.logpowerspec import logpowspec_multichannel

import kaldi_io


def build_from_path(wavlist, out_dir, rstfilename, stft_conf, num_workers=1):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    utt_list = []
    for wav_info in wavlist:
        items = wav_info.split()
        utt_idx = items[0]
        utt_list.append(utt_idx)
        channel = 0
        wav_path = items[1]
        futures.append(executor.submit(
            partial(_process_utterance, channel, wav_path, stft_conf)))

    print('extract features done, writing results...')

    # j = 0
    # for future in futures:
    #     if j%10 == 0:
    #         print(j)
    #     j += 1
    #     print(future.result()[0])
    ark_scp_output='ark:| copy-feats ark:- ark,scp:'+out_dir+'/'+rstfilename+'.ark,'+out_dir+'/'+rstfilename+'.scp'
    write_matrix(utt_list, [future.result() for future in futures], ark_scp_output)
    # return [future.result() for future in tqdm(futures)]

def _process_utterance(channel, wav_path, stft_conf):
    lps = logpowspec_multichannel(wav_path, channel, sr=stft_conf['sample_rate'], n_fft=stft_conf['n_fft'], 
        hop_length=stft_conf['hop_length'], win_length=stft_conf['win_length'], window=stft_conf['window'], pre_emphasis=stft_conf['pre_emphasis'])
    # lps_filename = os.path.join(out_dir, utt_idx+".npy")
    # np.save(lps_filename, lps.astype(np.float32), allow_pickle=False)
    # return lps_filename
    return lps


def preprocess(wavlist, out_dir, rstfilename, stft_conf, nj):
    os.makedirs(out_dir, exist_ok=True)
    build_from_path(wavlist, out_dir, rstfilename, stft_conf, nj)
    # print(metadata)
    # write_metadata(metadata, out_dir)

def write_matrix(utt_list, matrix_list, filename):
	with kaldi_io.open_or_fd(filename, 'wb') as f:
		for key, matrix in zip(utt_list, matrix_list):
			kaldi_io.write_mat(f, matrix, key=key)

# def write_metadata(metadata, out_dir):
#     with open(os.path.join(os.path.dirname(out_dir), 'log'), 'w', encoding='utf-8') as f:
#         for m in metadata:
#             f.write('|'.join([str(x) for x in m]) + '\n')
#     samples = sum([m[2] for m in metadata])
#     sr = hparams.sample_rate
#     hours = samples / sr / 3600
#     print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), samples, hours))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-wav-file', type=str, default='/scratch/xli/kaldi/egs/voxceleb_xli/v1/data/voxceleb1_train/wav.scp')
    parser.add_argument('--test-wav-file', type=str, default='/scratch/xli/kaldi/egs/voxceleb_xli/v1/data/voxceleb1_test/wav.scp')
    parser.add_argument('--num-workers', type=int, default=5)
    parser.add_argument('--out-dir', type=str, default='./data/LPMS')
    parser.add_argument('--train-rstfilename', type=str, default='train')
    parser.add_argument('--test-rstfilename', type=str, default='test')
    parser.add_argument('--param-json-path', type=str, default='./conf/stft.json')
    args = parser.parse_args()

    args.num_workers = 5
    print("number of workers: ", args.num_workers)


    # extract LPMS
    with codecs.open(args.param_json_path, 'r', encoding='utf-8') as f:
        stft_conf = json.load(f)
    print(stft_conf)

    print("Preprocess train data ...")
    rfile = open(args.train_wav_file, 'r')
    wavlist = rfile.readlines()
    rfile.close()
    preprocess(wavlist, args.out_dir, args.train_rstfilename, stft_conf, args.num_workers)

    print("Preprocess test data ...")
    rfile = open(args.test_wav_file, 'r')
    wavlist = rfile.readlines()
    rfile.close()
    preprocess(wavlist, args.out_dir, args.test_rstfilename, stft_conf, args.num_workers)

    print("DONE!")
    sys.exit(0)
