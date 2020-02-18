##
spoofed_featsfile = 'data/voxceleb1_test/spoofed_target_voiced_feats_sigma0.3.scp'
ori_vadfile = 'data/voxceleb1_test/vad.scp'

testid2vad = {}
with open(ori_vadfile, 'r') as f:
	for line in f.readlines():
		testid, vad = line.split()
		testid2vad.update({testid:vad})

wfile = open('data/voxceleb1_test/spoofed_vad_target.scp', 'w')

with open(spoofed_featsfile, 'r') as f:
	for line in f.readlines():
		trialsid = line.split()[0]
		testid = trialsid[26:]
		wfile.write('%s %s\n' %(trialsid, testid2vad.get(testid)))

wfile.close()
