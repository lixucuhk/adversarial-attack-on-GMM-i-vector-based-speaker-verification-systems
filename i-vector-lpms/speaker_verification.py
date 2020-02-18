##
import time

import kaldi_io
import torch
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial

from local.gmm import FullGMM
from local.ivector_extract import ivectorExtractor
from local.data_prepare import load_data
from local.plda import PLDA

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sigma', default="1.0", type=str)
parser.add_argument('--datafile', default='data/voxceleb1_test/spoofed_voiced_feats_sigma1.0.scp', type=str)
parser.add_argument('--fgmmfile', type=str, default='exp/full_ubm/final_ubm.txt')
parser.add_argument('--ivector-extractor', type=str, default='exp/extractor/final_ie.txt')
parser.add_argument('--plda-mdl', type=str, default='exp/ivectors_train/plda.txt')
parser.add_argument('--enroll-ivectors', type=str, default='exp/ivectors_voxceleb1_test/enrollivectors.scp')
parser.add_argument('--trials', type=str, default='data/voxceleb1_test/trials_adv')
parser.add_argument('--ivector-global-mean', type=str, default='exp/ivectors_train/mean.vec')

parser.add_argument('--expt-rstfile', type=str, default='exp/verification_scores_sigma')

args = parser.parse_args()

start_time = time.time()

sigma = args.sigma
fgmmfile = args.fgmmfile
datafile = args.datafile
ivectorfile = args.ivector_extractor
pldamdlfile = args.plda_mdl
enrollscpfile = args.enroll_ivectors
trialsfile = args.trials
ivector_global_meanfilename = args.ivector_global_mean
expt_rstfile = '%s%s' %(args.expt_rstfile, sigma)

fgmm = FullGMM(fgmmfile)
time1 = time.time()
print('Loading fgmm model complete. %d s' %(time1-start_time))

extractor = ivectorExtractor(ivectorfile)
ivector_dim = extractor.ivector_dim
time2 = time.time()
print('Loading ivectorextractor complete. %d s' %(time2-time1))

voiced_feats = load_data(datafile)
time3 = time.time()
print('Loading data complete. %d s' %(time3-time2))

ivector_global_meanfile = open(ivector_global_meanfilename, 'r')
line = ivector_global_meanfile.readline()
data = line.split()[1:-1]
for i in range(extractor.ivector_dim):
	data[i] = float(data[i])
ivector_global_mean = torch.tensor(data)
ivector_global_meanfile.close()

num_data = len(voiced_feats.data)
ivector_data = []
for i in range(num_data):
	data = torch.tensor(voiced_feats.data[i])
	zeroth_stats, first_stats = fgmm.Zeroth_First_Stats(data)
	ivector, L_inv, linear = extractor.Extractivector(zeroth_stats, first_stats)
	# ivector = extractor.LengthNormalization(ivector, torch.sqrt(torch.tensor(ivector_dim, dtype=torch.float)))
	ivector = extractor.SubtractGlobalMean(ivector, ivector_global_mean)
	ivector = extractor.LengthNormalization(ivector, torch.sqrt(torch.tensor(ivector_dim, dtype=torch.float)))
	ivector_data.append(ivector)

time4 = time.time()
print('Totally %d utts. Extracting ivectors complete. %d s' %(num_data, time4-time3))

plda_mdl = PLDA(pldamdlfile)
enrollkeys, enrollivectors = plda_mdl.ReadIvectors(enrollscpfile)
enrollivectors = torch.tensor(enrollivectors)

for i in range(num_data):
	ivector_data[i] = plda_mdl.TransformIvector(ivector_data[i], 1, simple_length_norm=False, normalize_length=True)

for i in range(len(enrollkeys)):
	enrollivectors[i] = plda_mdl.TransformIvector(enrollivectors[i], 1, simple_length_norm=False, normalize_length=True)

time41 = time.time()
print('Transform ivectors. %d s' %(time41-time4))

rfile = open(trialsfile, 'r')
wfile = open(expt_rstfile, 'w')
for line in rfile.readlines():
	enrollid, testid = line.split()[:2]
	enrollindex = enrollkeys.index(enrollid)
	testindex = voiced_feats.keys.index(testid)
	enrollivector = enrollivectors[enrollindex]
	testivector = ivector_data[testindex]
	score = plda_mdl.ComputeScores(enrollivector, 1, testivector)
	wfile.write('%s %s %f\n' %(enrollid, testid, score))

rfile.close()
wfile.close()
time5 = time.time()
print('PLDA scoring complete. %d s' %(time5-time4))

