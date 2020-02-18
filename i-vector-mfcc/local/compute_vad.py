##
import kaldi_io
import numpy as np
import sys

dataset = sys.argv[1]

rfilename = '/scratch/xli/kaldi/egs/voxceleb_xli/v3_stft/data/voxceleb1_'+dataset+'/feats.scp'
ark_scp_output='ark:| copy-vector ark:- ark,scp:data/LPS/'+dataset+'_vad.ark,data/voxceleb1_'+dataset+'/vad.scp'

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

