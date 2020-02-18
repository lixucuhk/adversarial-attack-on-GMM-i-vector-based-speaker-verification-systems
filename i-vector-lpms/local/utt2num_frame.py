##
import kaldi_io
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--vad-scp', type=str, default='data/voxceleb1_test/vad.scp')
parser.add_argument('--utt2num_frames', type=str, default='data/voxceleb1_test/utt2num_frames')

args = parser.parse_args()

rfilename = args.vad_scp
wfilename = args.utt2num_frames

wfile = open(wfilename, 'w')

for key, vec in kaldi_io.read_vec_flt_scp(rfilename):
	wfile.write('%s %d \n' %(key, int(vec.sum())))

wfile.close()
