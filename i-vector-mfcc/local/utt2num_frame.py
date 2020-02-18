##
import kaldi_io
import numpy as np
import sys

dataset = sys.argv[1]

rfilename = '/scratch/xli/kaldi/egs/voxceleb_xli/v3_stft/data/'+dataset+'/vad.scp'
wfilename = '/scratch/xli/kaldi/egs/voxceleb_xli/v3_stft/data/'+dataset+'/utt2num_frames'

wfile = open(wfilename, 'w')

for key, vec in kaldi_io.read_vec_flt_scp(rfilename):
	wfile.write('%s %d \n' %(key, int(vec.sum())))

wfile.close()
