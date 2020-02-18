##
import torch
import torch.nn.functional as F

from local.data_prepare import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

class FullGMM(object):
	def __init__(self, mdlfile, random=False):
		if random == True:
			self.num_gaussians = 2048
			self.dim = 60
			self.gconsts = torch.ones(self.num_gaussians, device=device)
			self.weights = torch.ones(self.num_gaussians, device=device)
			self.means_invcovars = torch.ones(self.num_gaussians, self.dim, device=device)
			self.invcovars = torch.ones(self.num_gaussians, self.dim, self.dim, device=device)
		else:
			rdfile = open(mdlfile, 'r')
			line = rdfile.readline()
			while line != '':
				if '<GCONSTS>' in line:
					print('processing <GCONSTS>')
					gconsts = line.split()[2:-1]
					self.num_gaussians = len(gconsts)
					for i in range(self.num_gaussians):
						gconsts[i] = float(gconsts[i])
					self.gconsts = torch.tensor(gconsts, device=device)
					line = rdfile.readline()
				elif '<WEIGHTS>' in line:
					print('processing <WEIGHTS>')
					weights = line.split()[2:-1]
					# if len(weights) != self.num_gaussians:
					# 	print('Dimension does not match between weights and gconsts.')
					# 	exit(1)
					for i in range(self.num_gaussians):
						weights[i] = float(weights[i])
					self.weights = torch.tensor(weights, device=device)
					line = rdfile.readline()
				elif '<MEANS_INVCOVARS>' in line:
					print('processing <MEANS_INVCOVARS>')
					line = rdfile.readline()
					means_invcovars = []
					for i in range(self.num_gaussians):
						data = line.split(' ')[2:-1]
						for j in range(len(data)):
							data[j] = float(data[j])
						means_invcovars.append(data)
						line = rdfile.readline()
					self.dim = len(data)
					self.means_invcovars = torch.tensor(means_invcovars, device=device)            # (self.num_gaussians, self.dim)
					print(self.means_invcovars.size())
				elif '<INV_COVARS>' in line:
					print('processing <INV_COVARS>')
					self.invcovars = torch.zeros(self.num_gaussians, self.dim, self.dim, device=device)
					for i in range(self.num_gaussians):
						line = rdfile.readline()
						for j in range(self.dim):
							data = line.split(' ')[:-1]
							for k in range(len(data)):
								self.invcovars[i][j][k] = float(data[k])
								self.invcovars[i][k][j] = float(data[k])
							line = rdfile.readline()
					# for i in range(self.num_gaussians):
					# 	self.invcovars[i] = self.SymmetricMatrix(self.invcovars[i])
				else:
					line = rdfile.readline()
			rdfile.close()
		self.Means() # (self.num_gaussians, self.dim)

	def Means(self):
		print('processing <Means>')
		self.means = torch.zeros(self.num_gaussians, self.dim, device=device)
		self.means = torch.matmul(torch.inverse(self.invcovars), self.means_invcovars.unsqueeze(-1)).squeeze(-1)
		print(self.means.size())


	def SymmetricMatrix(self, matrix):
		num_row, num_col = matrix.size()
		new_matrix = matrix
		for i in range(num_row):
			for j in range(i+1, num_col):
				new_matrix[i][j] = matrix[j][i]

		return new_matrix

	def ComponentLogLikelihood(self, data):
		# loglike = torch.zeros(self.num_gaussians)
		loglike = torch.matmul(self.means_invcovars, data)
		# print('!!!!!!!!!!!!!!!!!!')
		# print(loglike)
		loglike -= 0.5*torch.matmul(torch.matmul(self.invcovars, data), data)
		# print('!!!!!!!!!!!!!!!!!!')
		# print(loglike)
		loglike += self.gconsts
		# print('!!!!!!!!!!!!!!!!!!')
		# print(loglike)
		# print('!!!!!!!!!!!!!!!!!!')

		return loglike

	def Posterior(self, data):
		post = F.softmax(self.ComponentLogLikelihood(data), -1)

		return post

	def Zeroth_FirstCenter_Stats(self, data_seq):
		num_frame = len(data_seq)
		zeroth_stats = torch.zeros(self.num_gaussians, device=device)
		firstcenter_stats = torch.zeros(self.num_gaussians, self.dim, device=device)
		for i in range(num_frame):
			post = self.Posterior(data_seq[i])
			zeroth_stats += post
			firstcenter_stats += torch.mm(post.unsqueeze(-1), data_seq[i].unsqueeze(0))

		firstcenter_stats -= torch.mm(torch.diag(zeroth_stats), self.means)

		return zeroth_stats, firstcenter_stats

	def Zeroth_First_Stats(self, data_seq):
		num_frame = len(data_seq)
		zeroth_stats = torch.zeros(self.num_gaussians, device=device)
		first_stats = torch.zeros(self.num_gaussians, self.dim, device=device)
		for i in range(num_frame):
			post = self.Posterior(data_seq[i])
			zeroth_stats += post
			first_stats += torch.mm(post.unsqueeze(-1), data_seq[i].unsqueeze(0))

		# firstcenter_stats -= torch.mm(torch.diag(zeroth_stats), self.means)

		return zeroth_stats, first_stats

	def DRV_Ni(self, data, post):
		centered_data = -1*self.means+data # C*F
		sig_in_cent = torch.matmul(self.invcovars, centered_data.unsqueeze(-1)).squeeze(-1) # C*F
		const = torch.matmul(sig_in_cent.t(), post).unsqueeze(-1) # F*1
		Ni_drv = torch.matmul(const, post.unsqueeze(-1).t())-torch.matmul(sig_in_cent.t(), torch.diag(post))

		return Ni_drv

	def DRV_ui(self, Ni_drv, data, post):
		ui_drv = []
		for i in range(self.num_gaussians):
			drv = post[i]*torch.eye(self.dim, device=device)+torch.matmul(Ni_drv[:,i].unsqueeze(-1), data.unsqueeze(-1).t())
			ui_drv.append(drv)
		ui_drv = torch.cat(ui_drv, 1) # F*CF

		return ui_drv

	def ui(self, posts_seq, data_seq):
		ui = []
		for i in rang(self.num_gaussians):
			uij = torch.matmul(data_seq.t(), post_seq[i]) # F*1
			ui.append(uij.t())

		ui = torch.cat(ui, 0)

		return ui

	def post_seq(self, data_seq):
		posts = []
		for i in range(len(data_seq)):
			posts.append(self.Posterior(data_seq[i]))

		posts = torch.stack(posts, 0).t()

		return posts


if __name__ == '__main__':
	fgmmfile = '/scratch/xli/kaldi/egs/sre10_ori/v1/exp/full_ubm_2048/final_ubm.txt'
	datafile = 'data/sre10_test/voiced_feats_3.scp'

	fgmm = FullGMM(fgmmfile)
	gconsts = fgmm.gconsts
	means_invcovars = fgmm.means_invcovars
	invcovars = fgmm.invcovars
	weights = fgmm.weights

	# print(gconsts[0])
	# print(means_invcovars[0])
	# print(invcovars[0])
	# print(weights[0])
	print('Loading fgmm model complete.')

	voiced_feats = load_data(datafile)
	print('Loading data complete.')

	data = torch.tensor(voiced_feats.data[0])
	key = voiced_feats.keys[0]

	post_seq = fgmm.post_seq(data)
	print(post_seq.size())
	# for i in range(6):
	# 	frame_data = data[i]
	# 	c_loglikelihood = fgmm.ComponentLogLikelihood(frame_data)
	# 	print(c_loglikelihood)
	# 	# c_likelihood = torch.exp(c_loglikelihood)
	# 	# print(c_likelihood)
	# 	# likelihood = torch.sum(c_likelihood)
	# 	loglikelihood = torch.logsumexp(c_loglikelihood, -1)
	# 	print('Frame %d of utt %s, loglikelihood: %f\n' %(i, key, loglikelihood))

	# for i in range(6):
	# 	frame_data = data[i]
	# 	posterior = fgmm.Posterior(frame_data)
	# 	print(posterior[620])
	# 	print(posterior[707])

	# zeroth_stats, first_stats = fgmm.Zeroth_FirstCenter_Stats(data)
	# print('\n\n\n')
	# print(zeroth_stats)
	# print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
	# print(first_stats[0])
