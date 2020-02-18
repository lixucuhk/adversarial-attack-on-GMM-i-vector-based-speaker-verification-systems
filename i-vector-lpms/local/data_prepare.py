##
import torch
import kaldi_io
import numpy

class load_data(object):
	def __init__(self, data_scp_file):
		self.keys = []
		self.data = []
		i = 0
		for key, mat in kaldi_io.read_mat_scp(data_scp_file):
			# print(key)
			# print(mat)
			# print(len(mat.tolist()))
			# exit(0)
			i += 1
			self.keys.append(key)
			self.data.append(mat.tolist())
		print('totally %d utts' %(i))

	def GetTrialsdata(self, trialsfile):
		rfile = open(trialsfile, 'r')
		trials_data = []
		trials_info = []
		for line in rfile.readlines():
			enrollid, testid, groundtruth = line.split()
			index = self.keys.index(testid)
			trials_data.append(self.data[index])
			trials_info.append([enrollid, testid, groundtruth])
		rfile.close()

		return trials_data, trials_info

	
	# def GetKeysValues(self):
	# 	uttids = self.data.keys()
	# 	values = []
	# 	for uttid in uttids:
	# 		values.append(self.data.get(uttid))

	# 	return uttids, values



if __name__ == '__main__':
	datafile = 'data/sre10_test/voiced_feats.scp'
	voiced_feats = load_data(datafile)
	for i in range(1):
		print('Key is %s, value is \n  %s' %(voiced_feats.keys[0], voiced_feats.data[0][1]))
