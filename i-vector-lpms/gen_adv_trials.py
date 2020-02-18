##
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ori-trials', type=str, default='data/voxceleb1_test/trials')
parser.add_argument('--adv-trials', type=str, default='data/voxceleb1_test/trials_adv')

args = parser.parse_args()

rfile = open(args.ori_trials, 'r')
wfile = open(args.adv_trials, 'w')

for line in rfile.readlines():
	enrollid, testid, gt = line.split()
	# if gt == 'target':
	# 	wfile.write(line)
	# else:
	testid = enrollid+'_'+testid
	wfile.write('%s %s %s\n' %(enrollid, testid, gt))

rfile.close()
wfile.close()
