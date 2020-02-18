##
import time

import kaldi_io
import torch
import torch.nn.functional as F

from local.gmm import FullGMM
from local.ivector_extract import ivectorExtractor
from local.data_prepare import load_data
from local.plda import PLDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

class Derivator(object):
	def __init__(self, gmm, extractor, plda, ivector_meanfile):
		self.gmm = gmm
		self.extractor = extractor
		self.plda = plda
		self.T = self.extractor.T # CF*D
		self.sigma_inv = self.extractor.big_sigma_inv # CF*CF
		self.ivector_dim = self.extractor.ivector_dim # D
		rfile = open(ivector_meanfile, 'r')
		line = rfile.readline()
		data = line.split()[1:-1]
		for i in range(self.ivector_dim):
			data[i] = float(data[i])
		self.ivector_mean = torch.tensor(data, device=device)
		rfile.close()

	def Ivector_IvectorDerivation(self, acstc_data):
		print('Computing ivector derivation.')
		zeroth_stats, first_stats = self.gmm.Zeroth_First_Stats(acstc_data)
		ivector, L_inv, u = self.extractor.Extractivector(zeroth_stats, first_stats)

		post_seq = self.gmm.post_seq(acstc_data).t()
		# ui = self.gmm.ui(post_seq, acstc_data)
		# L_inv = self.extractor.L_inv(post_seq) # D*D
		# u = torch.matmul(torch.matmul(self.T.t(), self.sigma_inv), ui) # D*1
		# u[0] += self.extractor.offset
		U = torch.zeros(self.ivector_dim*self.ivector_dim, self.ivector_dim, device=device) # D^2*D
		# for i in range(self.ivector_dim):
		# 	U[i*self.ivector_dim:(i+1)*self.ivector_dim][i] = u

		x_drvs = []
		for i in range(len(acstc_data)):
			Ni_drv = self.gmm.DRV_Ni(acstc_data[i], post_seq[i]) # F*C
			ui_drv = self.gmm.DRV_ui(Ni_drv, acstc_data[i], post_seq[i]) # F*CF
			L_inv_drv = self.extractor.DRV_L_inv(Ni_drv, L_inv) # F*D^2
			u_drv = torch.matmul(torch.matmul(ui_drv, self.sigma_inv), self.T)

			x_drv = torch.matmul(L_inv_drv, U)+torch.matmul(u_drv, L_inv) ## F*D
			x_drvs.append(x_drv)

		x_drvs = torch.stack(x_drvs, 0) ## Ti*F*D

		return ivector, x_drvs

	def TransformIvector_TransformDerivation(self, ivector, expected_length, num_examples, simple_length_norm=False, normalize_length=True):
		print('Compute transform derivation.')
		norm_drv1 = self.extractor.DRV_norm(expected_length, ivector)
		ivector = self.extractor.LengthNormalization(ivector, expected_length)

		ivector = self.extractor.SubtractGlobalMean(ivector, self.ivector_mean)

		norm_drv2 = self.extractor.DRV_norm(expected_length, ivector)
		ivector = self.extractor.LengthNormalization(ivector, expected_length)

		transformivector_drv = self.plda.DRV_TransformIvector(ivector, num_examples, \
			                        simple_length_norm=simple_length_norm, normalize_length=normalize_length)
		trans_ivector = self.plda.TransformIvector(ivector, num_examples, \
			                        simple_length_norm=simple_length_norm, normalize_length=normalize_length)

		final_drv = torch.matmul(torch.matmul(norm_drv1, norm_drv2), transformivector_drv)

		return trans_ivector, final_drv


	def Score_ScoreDerivation(self, trans_enrollivector, trans_ivector, num_examples):
		score_drv = self.plda.DRV_scores(trans_enrollivector, num_examples, trans_ivector)
		score = self.plda.ComputeScores(trans_enrollivector, num_examples, trans_ivector)

		return score, score_drv
