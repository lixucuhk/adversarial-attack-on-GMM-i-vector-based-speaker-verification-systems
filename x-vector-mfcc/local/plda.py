##
import torch
import kaldi_io
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

class PLDA(object):
	def __init__(self, mdlfile, random=False):
		if random == True:
			self.dim = 600
			self.mean = torch.ones(self.dim, device=device)
			self.transform = torch.ones(self.dim, self.dim, device=device)
			self.psi = torch.ones(self.dim, device=device)
		else:
			rdfile = open(mdlfile, 'r')
			line = rdfile.readline()
			data = line.split()[2:-1]
			self.dim = len(data)
			for i in range(self.dim):
				data[i] = float(data[i])
			self.mean = torch.tensor(data, device=device)

			line = rdfile.readline()
			line = rdfile.readline()
			transform_matrix = []
			for i in range(self.dim):
				data = line.split(' ')[2:-1]
				for j in range(self.dim):
					data[j] = float(data[j])
				transform_matrix.append(data)
				line = rdfile.readline()
			self.transform = torch.tensor(transform_matrix, device=device)

			data = line.split()[1:-1]
			for i in range(self.dim):
				data[i] = float(data[i])
			self.psi = torch.tensor(data, device=device)

			rdfile.close()

	def ReadIvectors(self, ivectorfile):
		keys = []
		data = []
		i = 0
		for key, mat in kaldi_io.read_vec_flt_scp(ivectorfile):
			# print(key)
			# print(mat)
			# print(len(mat.tolist()))
			# exit(0)
			i += 1
			keys.append(key)
			data.append(mat.tolist())
		print('totally %d ivectors' %(i))
		return keys, data

	def TransformIvector(self, ivector, num_examples, simple_length_norm, normalize_length):
		trans_ivector = torch.matmul(self.transform, ivector-self.mean)
		factor = 1.0
		if simple_length_norm == True:
			factor = torch.sqrt(self.dim)/torch.norm(trans_ivector, 2)
		elif normalize_length == True:
			factor = self.GetNormalizaionFactor(trans_ivector, num_examples)

		# print('original ivector is \n')
		# print(trans_ivector)
		trans_ivector = trans_ivector*factor
		# print('factor is %f' %(factor))
		# print('transformed ivector is \n')
		# print(trans_ivector)

		return trans_ivector


	def GetNormalizaionFactor(self, trans_ivector, num_examples):
		trans_ivector_sq = torch.pow(trans_ivector, 2)
		inv_covar = 1.0/(self.psi + 1.0/num_examples)
		factor = torch.sqrt(self.dim / torch.dot(inv_covar, trans_ivector_sq))

		return factor

	def ComputeScores(self, trans_trainivector, num_examples, trans_testivector):
		# trans_trainivector = self.TransformIvector(trainivector, num_examples, simple_length_norm, normalize_length)
		# trans_testivector = self.TransformIvector(testivector, 1, simple_length_norm, normalize_length)

		#### work out loglike_given_class
		mean = torch.zeros(self.dim)
		variance = torch.zeros(self.dim)

		for i in range(self.dim):
			mean[i] = num_examples*self.psi[i]/(num_examples*self.psi[i]+1.0)*trans_trainivector[i]
			variance[i] = 1.0+self.psi[i]/(num_examples*self.psi[i]+1.0)

		logdet = torch.sum(torch.log(variance))

		sqdiff = torch.pow(trans_testivector-mean, 2)
		variance = 1.0/variance

		loglike_given_class = -0.5*(logdet + torch.log(2*torch.tensor(3.1415926, device=device))*self.dim + torch.dot(sqdiff, variance))

		### work out loglike_without_class
		sqdiff = torch.pow(trans_testivector, 2)
		variance = self.psi + 1.0
		logdet = torch.sum(torch.log(variance))
		variance = 1.0/variance
		loglike_without_class = -0.5*(logdet + torch.log(2*torch.tensor(3.1415926, device=device))*self.dim + torch.dot(sqdiff, variance))

		loglike_ratio = loglike_given_class - loglike_without_class

		return loglike_ratio

	def DRV_TransformIvector(self, ivector, num_examples, simple_length_norm, normalize_length):
		############ Currently we only consider simple_length_norm = False situation.
		if normalize_length == True:
			trans_ivector = torch.matmul(self.transform, ivector-self.mean)
			factor = 1.0
			factor = self.GetNormalizaionFactor(trans_ivector, num_examples)

			norm_drv = torch.zeros(self.dim, self.dim, device=device)
			trans_ivector_sq = torch.pow(trans_ivector, 2)

			common_vector = torch.matmul(torch.diag(num_examples/(num_examples*self.psi+1)), \
				                          -1*trans_ivector_sq*torch.pow(factor, 3)/self.dim)

			for i in range(self.dim):
				norm_drv[:,i] += common_vector
				norm_drv[i][i] += factor

			transform_drv = torch.matmul(self.transform.t(), norm_drv)
		else:
			transform_drv = self.transform.t()

		return transform_drv

	def DRV_Scores(self, trans_trainivector, num_examples, trans_testivector):
		mean = torch.zeros(self.dim)
		v1 = torch.zeros(self.dim)

		for i in range(self.dim):
			mean[i] = num_examples*self.psi[i]/(num_examples*self.psi[i]+1.0)*trans_trainivector[i]
			v1[i] = 1.0+self.psi[i]/(num_examples*self.psi[i]+1.0)

		v1 = 1.0/v1
		v2 = 1.0/(1+self.psi)

		score_drv = torch.matmul(torch.diag(trans_testivector), v2)-torch.matmul(torch.diag(trans_testivector-mean), v1)

		return score_drv


if __name__ == '__main__':
	start_time = time.time()
	fgmmfile = '/scratch/xli/kaldi/egs/sre10_ori/v1/exp/full_ubm_2048/final_ubm.txt'
	datafile = 'data/sre10_test/voiced_feats_3.scp'
	ivectorfile = '/scratch/xli/kaldi/egs/sre10_ori/v1/exp/extractor/extractor.txt'
	pldamdlfile = '/scratch/xli/kaldi/egs/sre10_ori/v1/exp/ivectors_sre/plda.txt'
	enrollscpfile = '/scratch/xli/kaldi/egs/sre10_ori/v1/exp/ivectors_sre10_train/enrollivectors.scp'
	testscpfile = '/scratch/xli/kaldi/egs/sre10_ori/v1/exp/ivectors_sre10_test/testivectors.scp'

	# extractor = ivectorExtractor(ivectorfile)
	# sigma_inv = extractor.sigma_inv
	# extractor_matrix = extractor.extractor_matrix
	# time1 = time.time()
	# print('Loading ivectorextractor complete. %d s' %(time1-start_time))
	# # print(sigma_inv.size())
	# # print(sigma_inv[1])
	# # print(extractor_matrix.size())
	# # print(extractor_matrix[1])

	# fgmm = FullGMM(fgmmfile)
	# gconsts = fgmm.gconsts
	# means_invcovars = fgmm.means_invcovars
	# invcovars = fgmm.invcovars
	# weights = fgmm.weights
	# time2 = time.time()

	# print('Loading fgmm model complete. %d s' %(time2-time1))

	# voiced_feats = load_data(datafile)
	# time3 = time.time()
	# print('Loading data complete. %d s' %(time3-time2))

	# data = torch.tensor(voiced_feats.data[0])
	# key = voiced_feats.keys[0]
	# zeroth_stats, first_stats = fgmm.Zeroth_First_Stats(data)

	# ivectors = extractor.Extractivector(zeroth_stats, first_stats)
	# time4 = time.time()
	# print('Extracting ivectors complete. %d s' %(time4-time3))

	plda_mdl = PLDA(pldamdlfile)

	# print('###########################')
	# print(plda_mdl.mean)
	# print('###########################')
	# print(plda_mdl.transform)
	# print('###########################')
	# print(plda_mdl.psi)
	# print('###########################')
	# print(plda_mdl.dim)
	# print('###########################')
	enrollkeys, enrollivectors = plda_mdl.ReadIvectors(enrollscpfile)
	testkeys, testivectors = plda_mdl.ReadIvectors(testscpfile)
	enrollivectors = torch.tensor(enrollivectors)
	testivectors = torch.tensor(testivectors)

	# transformed_testivectors = plda_mdl.TransformIvector(testivectors[0], 1, simple_length_norm=False, normalize_length=True)
	# print(transformed_testivectors)

	score = plda_mdl.ComputeScores(enrollivectors[2706], 1, testivectors[64], normalize_length=True)
	print('enroll %s, test %s, score %f.' %(enrollkeys[2706], testkeys[64], score))
	score = plda_mdl.ComputeScores(enrollivectors[2706], 1, testivectors[304], normalize_length=True)
	print('enroll %s, test %s, score %f.' %(enrollkeys[2706], testkeys[304], score))
	score = plda_mdl.ComputeScores(enrollivectors[2706], 1, testivectors[309], normalize_length=True)
	print('enroll %s, test %s, score %f.' %(enrollkeys[2706], testkeys[309], score))
	score = plda_mdl.ComputeScores(enrollivectors[2706], 1, testivectors[324], normalize_length=True)
	print('enroll %s, test %s, score %f.' %(enrollkeys[2706], testkeys[324], score))
