##
import random

rfile = open('/scratch/xli/kaldi/egs/sre10_ori/v1/data/sre10_test/voiced_feats.scp', 'r')
wfile = open('/scratch/xli/kaldi/egs/sre10_ori/v1/data/sre10_test/voiced_feats_shuffled.scp', 'w')

lines = rfile.readlines()
random.shuffle(lines)
for line in lines:
	wfile.write(line)

rfile.close()
wfile.close()
