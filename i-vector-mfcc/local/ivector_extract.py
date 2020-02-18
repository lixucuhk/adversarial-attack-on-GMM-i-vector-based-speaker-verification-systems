##

import torch
from local.gmm import FullGMM
from local.data_prepare import load_data
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

class ivectorExtractor(object):
	def __init__(self, mdlfile, random=False):
		if random == True:
			self.num_gaussian = 2048
			self.dim = 60
			self.ivector_dim = 600
			self.extractor_matrix = torch.ones(self.num_gaussian, self.dim, self.ivector_dim, device=device)
			self.sigma_inv = torch.ones(self.num_gaussian, self.dim, self.dim, device=device)
			self.offset = torch.tensor(1.0, device=device)
		else:
			rdfile = open(mdlfile, 'r')
			line = rdfile.readline()
			while line != '':
				if '<w_vec>' in line:
					data = line.split()[2:-1]
					self.num_gaussian = len(data)
					line = rdfile.readline()
				elif '<M>' in line:
					extractor_matrix = []
					for i in range(self.num_gaussian):
						line = rdfile.readline()
						component_extractor_matrix = []
						while ']' not in line:
							data = line.split()
							for j in range(len(data)):
								data[j] = float(data[j])
							component_extractor_matrix.append(data)
							line = rdfile.readline()
						data = line.split()[:-1]
						for j in range(len(data)):
							data[j] = float(data[j])
						component_extractor_matrix.append(data)
						line = rdfile.readline()
						extractor_matrix.append(component_extractor_matrix)
					self.extractor_matrix = torch.tensor(extractor_matrix, device=device) # C*F*D
				elif '<SigmaInv>' in line:
					self.dim = self.extractor_matrix.size()[1]
					self.ivector_dim = self.extractor_matrix.size()[2]
					self.sigma_inv = torch.zeros(self.num_gaussian, self.dim, self.dim, device=device)
					for i in range(self.num_gaussian):
						line = rdfile.readline()
						for j in range(self.dim):
							data = line.split()
							for k in range(j+1):
								self.sigma_inv[i][j][k] = float(data[k])
								self.sigma_inv[i][k][j] = float(data[k])
							line = rdfile.readline()
				elif '<IvectorOffset>' in line:
					self.offset = torch.tensor(float(line.split()[1]), device=device)
					line = rdfile.readline()
				else:
					line = rdfile.readline()
			rdfile.close()

		# self.T = self.extractor_matrix.view(-1, self.ivector_dim) # CF*D
		# self.big_sigma_inv = torch.zeros(self.num_gaussian*self.dim, self.num_gaussian*self.dim, device=device)
		# for i in range(self.num_gaussian):
		# 	self.big_sigma_inv[i*self.dim:(i+1)*self.dim, i*self.dim:(i+1)*self.dim] = self.sigma_inv[i]

	def Extractivector(self, zeroth_stats, first_stats):
		L = torch.eye(self.ivector_dim, device=device)
		linear = torch.zeros(self.ivector_dim, device=device)

		for i in range(self.num_gaussian):
			L += zeroth_stats[i]*torch.matmul(torch.matmul(self.extractor_matrix[i].t(), self.sigma_inv[i]), self.extractor_matrix[i])
			linear += torch.matmul(torch.matmul(self.extractor_matrix[i].t(), self.sigma_inv[i]), first_stats[i])

		linear[0] += self.offset
		L_inv = torch.inverse(L)

		ivector = torch.matmul(L_inv, linear)
		ivector[0] -= self.offset

		return ivector, L_inv, linear

	def LengthNormalization(self, ivector, expected_length):
		input_norm = torch.norm(ivector)
		if input_norm == 0:
			print('Zero ivector!')
			exit(0)
		radio = expected_length/input_norm
		ivector = ivector*radio

		return ivector

	def SubtractGlobalMean(self, ivector, mean):
		ivector = ivector-mean
		return ivector

	def DRV_L_inv(self, Ni_drv, L_inv):
		# T = self.extractor_matrix.view(-1, self.ivector_dim) # CF*D
		# sigma_inv = torch.zeros(self.num_gaussian*self.dim, self.num_gaussian*self.dim, device=device)
		# for i in range(self.num_gaussian):
		# 	sigma_inv[i*self.dim:(i+1)*self.dim][i*self.dim:(i+1)*self.dim] = self.sigma_inv[i]

		const_left = -1*torch.matmul(torch.matmul(L_inv, self.T.t()), self.big_sigma_inv) # D*CF
		const_right = torch.matmul(self.T, L_inv) # CF*D

		L_inv_drv = []
		for i in range(self.dim):
			diag_elements = torch.matmul(torch.diag(Ni_drv[i]), torch.ones(self.num_gaussian, self.dim, device=device)).view(1, -1) # 1*CF
			diag_matrix = torch.diag(diag_elements.squeeze(0)) # CF*CF
			rst = torch.matmul(torch.matmul(const_left, diag_matrix), const_right)
			L_inv_drv.append(rst.view(1, -1))

		L_inv_drv = torch.cat(L_inv_drv, 0)

		return L_inv_drv

	def L_inv(self, post_seq):
		post_seq = post_seq.t() # Ti*C
		# T = self.extractor_matrix.view(-1, self.ivector_dim) # CF*D
		# sigma_inv = torch.zeros(self.num_gaussian*self.dim, self.num_gaussian*self.dim, device=device)
		# for i in range(self.num_gaussian):
		# 	sigma_inv[i*self.dim:(i+1)*self.dim][i*self.dim:(i+1)*self.dim] = self.sigma_inv[i]

		Ni_vec = torch.zeros(self.num_gaussians, device=device)
		for i in range(len(post_seq)):
			Ni_vec += post_seq[i]

		Ni = torch.diag(torch.matmul(torch.diag(Ni_vec), torch.ones(self.num_gaussian, self.num_gaussian, device=device)).view(1, -1))
		rightside = torch.matmul(torch.matmul(torch.matmul(self.T.t(), self.big_sigma_inv), Ni), self.T)

		L_inv = torch.inverse(torch.eye(self.ivector_dim, device=device)+rightside)

		return L_inv

	def DRV_norm(self, expected_length, vector):
		v_norm = torch.norm(vector)
		dim = vector.size()[-1]
		vec_sq = torch.pow(vector, 2)
		norm_drv = torch.zeros(dim, dim, device=device)

		common_vector = -1*vec_sq*expected_length*torch.pow(v_norm, -3)
		common_scalar = expected_length*torch.pow(v_norm, -1)
		for i in range(dim):
			norm_drv[:,i] += common_vector
			norm_drv[i][i] += common_scalar

		return norm_drv


if __name__ == '__main__':
	start_time = time.time()
	fgmmfile = '/scratch/xli/kaldi/egs/sre10_ori/v1/exp/full_ubm_2048/final_ubm.txt'
	datafile = 'data/sre10_test/voiced_feats_3.scp'
	ivectorfile = '/scratch/xli/kaldi/egs/sre10_ori/v1/exp/extractor/extractor.txt'

	extractor = ivectorExtractor(ivectorfile)
	sigma_inv = extractor.sigma_inv
	extractor_matrix = extractor.extractor_matrix
	time1 = time.time()
	print('Loading ivectorextractor complete. %d s' %(time1-start_time))
	# print(sigma_inv.size())
	# print(sigma_inv[1])
	# print(extractor_matrix.size())
	# print(extractor_matrix[1])

	fgmm = FullGMM(fgmmfile)
	gconsts = fgmm.gconsts
	means_invcovars = fgmm.means_invcovars
	invcovars = fgmm.invcovars
	weights = fgmm.weights
	time2 = time.time()

	print('Loading fgmm model complete. %d s' %(time2-time1))

	voiced_feats = load_data(datafile)
	time3 = time.time()
	print('Loading data complete. %d s' %(time3-time2))

	data = torch.tensor(voiced_feats.data[0])
	key = voiced_feats.keys[0]
	zeroth_stats, first_stats = fgmm.Zeroth_First_Stats(data)

	ivectors = extractor.Extractivector(zeroth_stats, first_stats)
	time4 = time.time()
	print('Extracting ivectors complete. %d s' %(time4-time3))
	print(ivectors)
