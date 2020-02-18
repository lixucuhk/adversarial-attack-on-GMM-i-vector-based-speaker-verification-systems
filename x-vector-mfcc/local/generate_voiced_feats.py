##
trialsfile = open('/scratch/xli/kaldi/egs/sre10_ori/v1/data/sre10_test/trials', 'r')
voiced_featsfile = open('/scratch/xli/kaldi/egs/sre10_ori/v1/data/sre10_test/voiced_feats_10.scp', 'r')
wfile = open('/scratch/xli/kaldi/egs/sre10_ori/v1/data/sre10_test/trials_10.scp', 'w')

testids = []
for line in voiced_featsfile.readlines():
	test = line.split()[0]
	if test not in testids:
		testids.append(test)

for line in trialsfile.readlines():
	enroll, test, gt = line.split()
	if test in testids:
		wfile.write(line)

trialsfile.close()
voiced_featsfile.close()
wfile.close()
