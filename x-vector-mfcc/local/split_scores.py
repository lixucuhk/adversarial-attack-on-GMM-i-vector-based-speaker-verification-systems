rfile = open('exp/plda_scores_python_spoofed_sigma0.3', 'r')
wfile1 = open('exp/plda_scores_python_spoofed_sigma0.3_ori', 'w')
wfile2 = open('exp/plda_scores_python_spoofed_sigma0.3_spoofed', 'w')

for line in rfile.readlines():
	if 'original' not in line:
		wfile1.write(line)
		wfile2.write(line)
	else:
		items = line.split()
		enroll = items[0]
		test = items[1]
		ori_score = items[4][:-1]
		spoofed_score = items[7]
		wfile1.write('%s %s %s\n' %(enroll, test, ori_score))
		wfile2.write('%s %s %s\n' %(enroll, test, spoofed_score))

rfile.close()
wfile1.close()
wfile2.close()
