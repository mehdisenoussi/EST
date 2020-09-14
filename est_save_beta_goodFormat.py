import glob
import nibabel as nib
import numpy as np
from scipy.io import loadmat
from pathlib import Path

################################################################
# Paths
hd_data_path = Path('/Volumes/mehdimac/ghent/est/data/')
betas_data_path = hd_data_path / 'holroyd2018/'
res_path = Path('./results/')
################################################################


obs_paths = glob.glob(betas_data_path.as_posix() + '/*')

# get the name of each observer
obs_codes = np.array([op.split('/')[-1] for op in obs_paths])
n_obs = len(obs_codes)

for obs_ind in np.arange(n_obs):
	print('obs %i' % obs_ind)
	print('\tload data...')
	obs_num = obs_ind + 1
	# array to store all betas
	n_betas = len(glob.glob((betas_data_path / ('%s/GLM/beta_0*'%obs_codes[obs_ind])).as_posix()))
	allbetas = np.zeros(shape = [n_betas, 64, 64, 33])
	all_nifti_obj = []

	for beta_ind in np.arange(n_betas):
		# file name of the beta
		filename = (betas_data_path /  ('%s/GLM/beta_0%03i.nii' % (obs_codes[obs_ind], beta_ind+1)))

		# load the beta 3D image
		all_nifti_obj.append(nib.load(filename.as_posix()))
		data = all_nifti_obj[-1].get_data()
		allbetas[beta_ind, ...] = data

	
	# load the mat file containing the information about each beta
	vbeta_file = glob.glob((res_path / ('Vbeta_*%02i.mat' % (obs_ind+1))).as_posix())[0]
	vbeta = loadmat(vbeta_file)['Vbeta'].squeeze()
	# to store whether a beta is in a coffee or tea sequence
	coffee_or_tea = np.zeros(n_betas, dtype = np.str)
	# store whether the beta is in a water 1st or water 2nd sequence
	water_order = np.zeros(n_betas, dtype = np.int)
	# store beta number
	beta_number = np.zeros(n_betas, dtype = np.int)
	# store which action it was
	action_n = np.zeros(n_betas, dtype = np.int)
	# store which block it was
	block = np.zeros(n_betas, dtype = np.int)
	# store whether this is a beta of interest (not a movement regressor or something like that)
	beta_of_interest_mask = np.zeros(n_betas, dtype = np.bool)
	for i in np.arange(n_betas):
		# check the length of the beta "title" to know whether it's a beta of interest and store it
		beta_of_interest_mask[i] = len(vbeta[i][5][0]) == 39
		# if beta of interest get the infos from vbeta (tea or coffee, etc.)
		if beta_of_interest_mask[i]:
			coffee_or_tea[i] = vbeta[i][5][0].split(') ')[-1][0]
			water_order[i] = np.int(vbeta[i][5][0].split(') ')[-1][2])
			action_n[i] = np.int(vbeta[i][5][0].split(') ')[-1][4])
		# else fill with null values
		else:
			coffee_or_tea[i] = 'R'
			water_order[i] = -1
			action_n[i] = -1
		# get the beta number
		beta_number[i] = np.int(vbeta[i][5][0].split('beta (')[1][:4])
		# get the block number
		block[i] = np.int(vbeta[i][5][0].split('Sn(')[1][0])

	# mask all variable and data of interest
	beta_of_int = allbetas[beta_of_interest_mask, ...]
	coffee_or_tea = coffee_or_tea[beta_of_interest_mask]
	water_order = water_order[beta_of_interest_mask]
	action_n = action_n[beta_of_interest_mask]
	block_of_int = block[beta_of_interest_mask]

	# get nifti affine
	func_affine = all_nifti_obj[0].affine

	np.savez((res_path / ('betas_and_affine_obs%02i.npz'\
		% int(obs_codes[obs_ind].split('_')[-1]))).as_posix(),
		{'beta_of_int':beta_of_int, 'coffee_or_tea':coffee_or_tea, 'water_order':water_order,
		'action_n':action_n, 'block':block_of_int,
		'func_affine': func_affine, 'func_shape':all_nifti_obj[0].shape})

