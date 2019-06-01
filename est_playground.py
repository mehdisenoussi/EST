import glob
import nibabel as nib
import numpy as np
from scipy.io import loadmat
from nilearn.image import new_img_like, load_img

################################################################
# make sure to change this path
################################################################
data_path = '/Volumes/MEHDIFAT/clayfmri/neuralData/'
################################################################


obs_paths = glob.glob(data_path + '*')

# get the name of each observer
obs_codes = np.array([op.split('/')[-1] for op in obs_paths])
n_obs = len(obs_codes)
# number of beta per observer
# n_betas = 124
scores_all = np.zeros(shape = [n_obs, 64, 64, 33])


# for obs_i in np.arange(11n_obs):
for obs_i in np.arange(11, 18):
	# obs_i += 11
	print('obs %i' % obs_i)
	print('\tload data...')
	# array to store all betas
	n_betas = len(glob.glob(data_path + obs_codes[obs_i] + '/GLM/beta_0*'))
	allbetas = np.zeros(shape = [n_betas, 64, 64, 33])
	all_nifti_obj = []

	for beta_ind in np.arange(n_betas):
		# file name of the beta
		filename = data_path + obs_codes[obs_i] + '/GLM/beta_0%03i.nii' % (beta_ind+1)
		# print(filename)
		# load the beta 3D image
		all_nifti_obj.append(nib.load(filename))
		data_nonans = all_nifti_obj[-1].get_data()
		# remove all NaNs and replace them with zeros
		data_nonans[np.isnan(data_nonans)] = 0
		allbetas[beta_ind, ...] = data_nonans
		# print('\tdone!')
	# load the "brain" mask
	mask = nib.load(data_path + obs_codes[obs_i] + '/GLM/mask.nii')


	# load the mat file containing the information about each beta
	vbeta_file = glob.glob('/Users/mehdi/work/ghent/danesh/code/Vbeta_*%02i.mat' % (obs_i+1))[0]
	vbeta = loadmat(vbeta_file)['Vbeta'].squeeze()
	# vbeta = loadmat('/Users/mehdi/work/ghent/danesh/code/Vbeta_SEQUENCE22JUL2016_%02i.mat' % )['Vbeta'].squeeze()
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


	# apply a mask for your data of interest
	# water pouring 1st versus second in coffee sequence

	### ONLY COFFEE WATER POURING
	classif_con = 'coffee_w1_vs_w2'

	if classif_con == 'coffee_w1_vs_w2':
		only_coffee_w1 = (coffee_or_tea == 'c') & (water_order == 1) & (action_n == 2)
		only_coffee_w2 = (coffee_or_tea == 'c') & (water_order == 2) & (action_n == 4)

		mask_classif = only_coffee_w1 | only_coffee_w2
		y = only_coffee_w1 + (only_coffee_w2*2) #np.zeros(len(mask_classif))
		y = y[mask_classif]

		beta_to_classif = beta_of_int[mask_classif, ...]



	# import the cross-validation procedure
	from sklearn.model_selection import KFold
	cv = KFold(n_splits=4)

	# import classification library
	import nilearn.decoding
	# create an IMG object from the array of betas
	X = new_img_like(all_nifti_obj[-1], data=np.rollaxis(beta_to_classif.T, -1).T)

	# y = pd.DataFrame(coffee_or_tea)
	# y = coffee_or_tea
	n_jobs = -1
	searchlight = nilearn.decoding.SearchLight(
	    mask,
	    process_mask_img=None,
	    radius=6, n_jobs=n_jobs,
	    verbose=50, cv=cv)
	print('\tdo classif...')
	searchlight.fit(X, y)
	scores_all[obs_i, ...] = searchlight.scores_
	res_img = new_img_like(all_nifti_obj[-1], data=scores_all[obs_i, ...])

	res_img.to_filename('obs%02i_classif_%s_res_sl6_image.nii' % (obs_i, classif_con))
	print('\tdone!\n\n\n')








