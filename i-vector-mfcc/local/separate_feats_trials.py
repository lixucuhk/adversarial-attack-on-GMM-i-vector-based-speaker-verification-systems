import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--voiced-feats', type=str, default='data/voxceleb1_test/voiced_feats.scp')
parser.add_argument('--trials', type=str, default='data/voxceleb1_test/trials')
parser.add_argument('--num-per-part', type=int, default=40)
args = parser.parse_args()

num_per_part = args.num_per_part
voiced_feats = args.voiced_feats
trials = args.trials

with open(voiced_feats, 'r') as feats_file:
	feats = feats_file.readlines()
	num_feats = len(feats)
	num_partation = int(num_feats/num_per_part) if num_feats%num_per_part == 0 else int(num_feats/num_per_part)+1
	print('In practice, split the trials into %d parts.' %(num_partation))

	for i in range(num_partation):
		sub_feats_name = '%s.%d.scp' %(voiced_feats[:-4], i)
		with open(sub_feats_name, 'w') as wfile:
			for j in range(num_per_part):
				num_line = i*num_per_part+j
				if num_line == num_feats:
					break
				wfile.write(feats[num_line])

with open(trials, 'r') as rfile:
	total_trials = rfile.readlines()

for i in range(num_partation):
	sub_feats_name = '%s.%d.scp' %(voiced_feats[:-4], i)
	sub_trials_name = '%s.%d' %(trials, i)

	with open(sub_feats_name, 'r') as rfile, open(sub_trials_name, 'w') as wfile:
		testids = []
		for line in rfile.readlines():
			test = line.split()[0]
			if test not in testids:
				testids.append(test)

		for trial in total_trials:
			enroll, test, gt = trial.split()
			if test in testids:
				wfile.write(trial)

print('Separate feats and the corresponding trials done.')
