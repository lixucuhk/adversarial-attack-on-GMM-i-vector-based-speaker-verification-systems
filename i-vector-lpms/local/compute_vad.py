##
import kaldi_io
import numpy as np
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--feats-scp', type=str, default='data/voxceleb1_test/feats.scp')
parser.add_argument('--vad-ark', type=str, default='data/LPMS/test_vad.ark')
parser.add_argument('--vad-scp', type=str, default='data/voxceleb1_test/vad.scp')
args = parser.parse_args()


rfilename = args.feats_scp
ark_scp_output = 'ark:| copy-vector ark:- ark,scp:'+args.vad_ark+','+args.vad_scp

# dataset = sys.argv[1]

# rfilename = '/scratch/xli/kaldi/egs/voxceleb_xli/v3_stft/data/voxceleb1_'+dataset+'/feats.scp'
# ark_scp_output='ark:| copy-vector ark:- ark,scp:data/LPS/'+dataset+'_vad.ark,data/voxceleb1_'+dataset+'/vad.scp'

keys = []
vectors = []

for key, mat in kaldi_io.read_mat_scp(rfilename):
	rst = []
	keys.append(key)
	num_frames, feats_dim = mat.shape
	for i in range(num_frames):
		if mat[i].sum() > feats_dim*(-60):
			rst.append(1.0)
		else:
			rst.append(0.0)
	rst = np.array(rst)	
	vectors.append(rst)

print('totally %d utts.' %(len(keys)))

with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f:
	for key, vec in zip(keys, vectors):
		kaldi_io.write_vec_flt(f, vec, key=key)

