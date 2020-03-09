##
import time
import sys
import argparse

import kaldi_io
import torch
import torch.nn.functional as F

from local.gmm import FullGMM
from local.ivector_extract import ivectorExtractor
from local.data_prepare import load_data
from local.plda import PLDA
from local.SVsystem import SVsystem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--fgmmfile', type=str, default='exp/full_ubm/final_ubm.txt')
parser.add_argument('--datafile', type=str, default='data/voxceleb1_test/part100/voiced_feats.scp')
parser.add_argument('--ivector-extractor', type=str, default='exp/extractor/final_ie.txt')
parser.add_argument('--plda-mdl', type=str, default='exp/ivectors_train/plda.txt')
parser.add_argument('--enroll-ivectors', type=str, default='exp/ivectors_voxceleb1_test/enrollivectors.scp')
parser.add_argument('--trials', type=str, default='data/voxceleb1_test/part100/trials')
parser.add_argument('--ivector-global-mean', type=str, default='exp/ivectors_train/mean.vec')
parser.add_argument('--gradfile-ark', type=str, default='data/voxceleb1_test/part100/signed_gradientfile.ark')
parser.add_argument('--gradfile-scp', type=str, default='data/voxceleb1_test/part100/signed_gradientfile.scp')
args = parser.parse_args()

def extract_test_ivector(testindex):
	ivector = SV_system.Getivector(acoustic_data[testindex])
	trans_ivector = SV_system.TransformIvectors(ivector, 1, simple_length_norm=False, normalize_length=True)

	return trans_ivector

def binary_search_index(index_sequence, targetid):
	low = 0
	high = len(index_sequence)-1
	while high >= low:
		mid = int((high+low)/2)
		if index_sequence[mid][1] > targetid:
			high = mid-1
		elif index_sequence[mid][1] < targetid:
			low = mid+1
		else:
			return index_sequence[mid][0]

	# cannot find the targetid
	return -1


time0 = time.time()

fgmmfile = args.fgmmfile
datafile = args.datafile
ivectorfile = args.ivector_extractor
pldamdlfile = args.plda_mdl
enrollscpfile = args.enroll_ivectors
trialsfile = args.trials
ivector_global_meanfilename = args.ivector_global_mean
gradfilename =  'ark:| copy-feats ark: ark,scp:%s,%s' %(args.gradfile_ark, args.gradfile_scp)


fgmm = FullGMM(fgmmfile)
extractor = ivectorExtractor(ivectorfile)
plda_mdl = PLDA(pldamdlfile)
SV_system = SVsystem(fgmm, extractor, plda_mdl, ivector_global_meanfilename)

ivector_dim = extractor.ivector_dim
data_dim = extractor.dim

time1 = time.time()
print('Loading models complete. %d s' %(time1-time0))

enrollkeys, enrollivectors = plda_mdl.ReadIvectors(enrollscpfile)
enrollivectors = torch.tensor(enrollivectors, device=device)

for i in range(len(enrollkeys)):
	enrollivectors[i] = SV_system.TransformIvectors(enrollivectors[i], 1, simple_length_norm=False, normalize_length=True)

time2 = time.time()
print('Loading enrollment i-vectors complete. %d s' %(time2-time1))

voiced_feats = load_data(datafile)
num_data = len(voiced_feats.data)
acoustic_data = []

for i in range(num_data):
	print('Doing %d utts.' %(i))
	data = torch.tensor(voiced_feats.data[i], requires_grad=True, device=device)
	acoustic_data.append(data)
time3 = time.time()
print('Loading testing acoustic features complete. %d s' %(time3-time2))

enrollkeys_index = list(enumerate(enrollkeys))
enrollkeys_index.sort(key=lambda x:x[1])
testkeys_index = list(enumerate(voiced_feats.keys))
testkeys_index.sort(key=lambda x:x[1])
time4 = time.time()
print('Sorting enrollment and testing keys for indexing. %d s' %(time4-time3))

rfile = open(trialsfile, 'r')

with kaldi_io.open_or_fd(gradfilename, 'wb') as grad_f:
	num_trials = 0
	for line in rfile.readlines():
# 		if num_trials < 10196:
# 			num_trials += 1
# 			continue
		enrollid, testid, groundtruth = line.split()
		print('enrollid %s, testid %s.' %(enrollid, testid))
		# enrollindex = enrollkeys.index(enrollid)
		enrollindex = binary_search_index(enrollkeys_index, enrollid)
		enrollivector = enrollivectors[enrollindex]
		# testindex = voiced_feats.keys.index(testid)
		testindex = binary_search_index(testkeys_index, testid)
		# testivector = ivector_data[testindex]
		testivector = extract_test_ivector(testindex)

		time30 = time.time()
		score = SV_system.ComputePLDAScore(enrollivector, testivector)
		time31 = time.time()
		if acoustic_data[testindex].grad is not None:
			acoustic_data[testindex].grad.zero_()

		score.backward(retain_graph=True)
		time32 = time.time()
		num_trials += 1
		print('Done %d trials, compute score %d s, gradient %d s.' %(num_trials, time31-time30, time32-time31))
		if groundtruth == 'target':
			grads = -1.0*torch.sign(acoustic_data[testindex].grad)
		else:
			grads = torch.sign(acoustic_data[testindex].grad)
		kaldi_io.write_mat(grad_f, grads.detach().numpy(), key=enrollid+'_'+testid)

rfile.close()


if __name__ == '__main__':
	## uncomment the codes above, and test the indexing procedure
	pldamdlfile = 'exp/ivectors_train/plda.txt'
	enrollscpfile = 'exp/ivectors_voxceleb1_test/enrollivector.scp'

	enrollid1 = 'id10270-5r0dWxy17C8-00001' ## the 1st
	enrollid2 = 'id10292-ya6VNZp-pXw-00032' ## the 2607th
	enrollid3 = 'id10309-vobW27_-JyQ-00015' ## the 4874th

	time0 = time.time()
	plda_mdl = PLDA(pldamdlfile)
	time1 = time.time()
	print('Loading models complete. %d s' %(time1-time0))

	enrollkeys, enrollivectors = plda_mdl.ReadIvectors(enrollscpfile)
	enrollivectors = torch.tensor(enrollivectors, device=device)
	time2 = time.time()
	print('Loading enrollment ivectors. %d s' %(time2-time1))

	enrollkeys_index = list(enumerate(enrollkeys))
	enrollkeys_index.sort(key=lambda x:x[1])
	time3 = time.time()
	print('Sorting enrollment keys. %d s' %(time3-time2))

	enrollindex = binary_search_index(enrollkeys_index, enrollid1)
	print('%s, %d' %(enrollid1, enrollindex))
	time40 = time.time()
	print('Indexing %d s' %(time40-time3))

	enrollindex = binary_search_index(enrollkeys_index, enrollid2)
	print('%s, %d' %(enrollid2, enrollindex))
	time41 = time.time()
	print('Indexing %d s' %(time41-time40))

	enrollindex = binary_search_index(enrollkeys_index, enrollid3)
	print('%s, %d' %(enrollid3, enrollindex))
	time42 = time.time()
	print('Indexing %d s' %(time42-time41))

