##
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ori-vad', type=str, default='data/voxceleb1_test/vad.scp')
parser.add_argument('--adv-feats', type=str, default='data/voxceleb1_test/spoofed_feats_sigma1.0.scp')
parser.add_argument('--adv-vad', type=str, default='data/voxceleb1_test/spoofed_vad.scp')
args = parser.parse_args()

spoofed_featsfile = args.adv_feats
ori_vadfile = args.ori_vad
spoofed_vadfile = args.adv_vad

testid2vad = {}
with open(ori_vadfile, 'r') as f:
	for line in f.readlines():
		testid, vad = line.split()
		testid2vad.update({testid:vad})

wfile = open(spoofed_vadfile, 'w')

with open(spoofed_featsfile, 'r') as f:
	for line in f.readlines():
		trialsid = line.split()[0]
		testid = trialsid[26:]
		wfile.write('%s %s\n' %(trialsid, testid2vad.get(testid)))

wfile.close()
