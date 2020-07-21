import glob, time
import nibabel as nib
import numpy as np
from scipy.io import loadmat
from nilearn.image import new_img_like, load_img
# import classification library
import nilearn.decoding
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy.linalg as npl
from nibabel.affines import apply_affine
# import the cross-validation procedure
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score


# function compute the average voxel value (in our case classification accuracies) in a anatomically defined ROI
def get_anat_roi_func_coord(func_data, anat_rois, anat_vox2func_vox, roi_inds, mask):
	# - func_data is a Nibabel Nifti image containing the functional data (e.g. GLM betas)
	# - anat_rois is a Nibabel Nifti image containing the anatomical ROIs
	# automatically segmented using Freesurfer
	# - anat_vox2func_vox is the transformation matrix for coordinates from
	# anatomical to functional (searchlight)
	# - roi_inds is a list of indices or codes referring to anatomical ROIs
	# in the anat_rois volume

	# Create a mask of the selected ROIs in anat space
	anat_roi_data = anat_rois.get_data().copy()
	temp_roi = np.zeros(anat_roi_data.shape, dtype=np.int)
	for roi_ind_n in roi_inds:
		temp_roi += (anat_rois.get_data() == roi_ind_n)

	# create a list of indices of the selected voxels (anat space)
	roi_mask = temp_roi.astype(np.bool)
	x, y, z = np.where(roi_mask)
	roi_maskxyz = np.vstack([x, y, z]).T
	# transform these indices in the functional space and approximate their location in functional space by turning them into integers
	roi_funcSpace_inds = apply_affine(anat_vox2func_vox, roi_maskxyz).astype(np.int)

	# create a volume in the functional space to extract the values of the selected ROI
	roi_volmask_funcSpace = np.zeros(shape = func_data.shape, dtype = np.bool)
	for roi_funcSpace_ind_n in roi_funcSpace_inds:
		roi_volmask_funcSpace[roi_funcSpace_ind_n[0], roi_funcSpace_ind_n[1], roi_funcSpace_ind_n[2]] = True

	roi_volmask_funcSpace = roi_volmask_funcSpace & mask

	# return the ROI mask indices in functional space, the ROI mask in volume in functional space
	return roi_funcSpace_inds, roi_volmask_funcSpace


################################################################
# make sure to change these paths
################################################################
# functional data paths
betas_data_path = '/Volumes/MEHDIFAT/misc/clayfmri/neuralData/'
local_data_path = '/Users/mehdi/work/ghent/side_projects/danesh/'

anat_data_path = '/Volumes/mehdimac/ghent/side_projects/danesh/'

freesurfer_path = '/Applications/freesurfer/'
################################################################


obs_paths = glob.glob(betas_data_path + '*')

# get the name of each observer
obs_codes = np.array([op.split('/')[-1] for op in obs_paths])
n_obs = len(obs_codes)

############# HOW MANY FEATURES TO KEEP #############
n_feat = 100


################################################################
# How many permutations do you want to do ??? (careful there..)
################################################################
n_perm = 10
################################################################

# One automatic FreeSurfer segementation
anat_rois = nib.load(anat_data_path + 'data/Ouput-Filesscanner/Sub01/mri/aparc+aseg.mgz')
# get the codes of the segmented areas in the chosen segmentation volume
uniqs_rois_ind = np.unique(anat_rois.get_data())

# get the labels associated with each number.
fs_labels = np.loadtxt(freesurfer_path + 'FreeSurferColorLUT.txt', dtype=np.str)
# create a dictionary to store all codes' labels
dict_code_label_all_seg = dict(zip(fs_labels[:, 0].astype(np.int), fs_labels[:, 1]))

# find the labels associated with the codes in our chosen segmentation volume
dict_code_label_this_seg = {}
for i in uniqs_rois_ind:
	dict_code_label_this_seg.update({i:fs_labels[fs_labels[:, 0] == np.str(i), 1]})

# ROIs
rois_inds = np.array([  [1026, 2026], # left and right rostral anterior cingulate cortex
						# [1025, 2025], # L&R entohirnal cx
						# [1016, 2016], # L&R parahippocampal cx
						# [17, 53],	  # L&R hippocampus
						[1002, 2002], # left and right caudal anterior cingulate cortex
						[1003, 2003], # left and right caudal middle frontal cortex
						[1018, 2018], # left and right pars opercularis
						[1019, 2019], # left and right pars orbitalis
						# [1014, 2014], # medialorbitofrontal
						[1012, 2012], # lateralorbitofrontal
						[1032, 2032], # frontal pole
						# [1011, 2011], # lateral occipital
						[1020, 2020]]) # left and right pars triangularis
n_rois = len(rois_inds)

classif_cons = ['cross_temporal_context_1', 'cross_subtask_context_1', 'cross_task_context_1'][:1]
n_contr = len(classif_cons)

scores_all = np.zeros(shape = [n_obs, n_rois, n_contr, 2, n_perm+1])

for obs_ind in np.arange(n_obs):
	print('obs %i' % obs_ind)
	print('\tload data...')
	obs_num = obs_ind + 1
	# array to store all betas
	n_betas = len(glob.glob(betas_data_path + obs_codes[obs_ind] + '/GLM/beta_0*'))
	allbetas = np.zeros(shape = [n_betas, 64, 64, 33])
	all_nifti_obj = []

	for beta_ind in np.arange(n_betas):
		# file name of the beta
		filename = betas_data_path + obs_codes[obs_ind] + '/GLM/beta_0%03i.nii' % (beta_ind+1)

		# load the beta 3D image
		all_nifti_obj.append(nib.load(filename))
		data = all_nifti_obj[-1].get_data()
		allbetas[beta_ind, ...] = data

	
	# load the mat file containing the information about each beta
	vbeta_file = glob.glob(local_data_path + 'code/Vbeta_*%02i.mat' % (obs_ind+1))[0]
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


	anat_rois = nib.load(anat_data_path + 'data/Ouput-Filesscanner/Sub%02i/mri/aparc+aseg.mgz' % (obs_ind+1))
	# only gray matter mask
	grayMat_mask = nib.load(local_data_path + 'results/masks/obs%02i_ribbon_grayMatter.nii' % obs_ind).get_data().astype(np.bool)

	# compute transformation of coordinates from anatomical to functional (searchlight volume)
	anat_vox2func_vox = npl.inv(all_nifti_obj[0].affine).dot(anat_rois.affine)

	for roi_ind, roi_n_inds in enumerate(rois_inds):
		print('\troi %i' % (roi_ind+1))
		roi_funcSpace_inds, roi_volmask_funcSpace =\
				get_anat_roi_func_coord(all_nifti_obj[0], anat_rois, anat_vox2func_vox, roi_n_inds, grayMat_mask)
		if roi_volmask_funcSpace.sum() < n_feat:
			print('\t\t!!!!!\t\tNOT ENOUGH VOXELS IN THIS ROI (%i voxels)\t\t!!!!!\n\n' % roi_volmask_funcSpace.sum())
			scores_all[obs_ind, roi_ind, :] = np.nan

		else:
			betas_roi_n = beta_of_int[:, roi_volmask_funcSpace]
			if np.any(np.all(np.logical_not(betas_roi_n.astype(np.bool)), axis=0)):
				print('\t\tthere are some zeros:\n\t\t\t%s\n\n' %\
					(np.str(np.argwhere(np.all(np.logical_not(betas_roi_n.astype(np.bool)), axis=0)).squeeze() )))

			### the different contrasts
			for contr_ind, classif_con in enumerate(classif_cons):
				t = time.time()
				print('\t\tcontrast: %s' % classif_con)
				if classif_con == 'cross_temporal_context_1':
					first_or_second_stir = (action_n == 3) | (action_n == 5)
					mask_classif = first_or_second_stir
					y = (action_n[mask_classif] == np.unique(action_n[mask_classif])[0]).astype(np.int)
					
					groups = (coffee_or_tea[mask_classif] == 't').astype(np.int)

				beta_to_classif = betas_roi_n[mask_classif, :]

				cv = LeaveOneGroupOut()
				# anova filter, take n_feat best ranked features
				anova_filter = SelectKBest(f_regression, k=n_feat)
				anova_svm = Pipeline([('scaler', StandardScaler()),
									('anova', anova_filter),
									('clf', svm.SVC(kernel = 'linear', max_iter = 3000))])

				scores_all[obs_ind, roi_ind, contr_ind, :, 0] =\
					cross_val_score(anova_svm, beta_to_classif, y, cv=cv,
						groups=groups, n_jobs = 10)

				for perm_n in np.arange(n_perm):
					np.random.shuffle(y)
					cv = LeaveOneGroupOut()
					# anova filter, take n_feat best ranked features
					anova_filter = SelectKBest(f_regression, k=n_feat)
					anova_svm = Pipeline([('scaler', StandardScaler()),
										('anova', anova_filter),
										('clf', svm.SVC(kernel = 'linear', max_iter = 3000))])

					scores_all[obs_ind, roi_ind, contr_ind, :, perm_n+1] =\
						cross_val_score(anova_svm, beta_to_classif, y, cv=cv,
							groups=groups, n_jobs = 10)











fig, axs = pl.subplots(1, 1)
# for roi_ind in np.arange(n_rois)
axs.errorbar(x = np.arange(n_rois),
	y=np.nanmean(np.nanmean(scores_all[...,0], axis=-1), axis=0),
	yerr=np.std(np.nanmean(scores_all[...,0], axis=-1), axis=0)/(n_obs**.5),
	fmt='o', zorder=0)
axs.set_ylim([.3, 1])
toplot = np.nanmean(scores_all[...,0].squeeze(), axis=-1)
t, p = stats.ttest_1samp(toplot, axis=0, popmean=.5)
axs.plot(np.arange(n_rois)[p<.05], np.nanmean(toplot, axis=0)[p<.05], 'ro', zorder=1)
axs.plot(np.arange(n_rois)[p<.05], np.nanmean(toplot, axis=0)[p<.01], 'yo', zorder=2)

perm_data = np.nanmean(scores_all[...,1:], axis=-2).squeeze()
# for roi_ind in np.arange(n_rois):
violins = axs.violinplot(perm_data.mean(axis=0).T, positions = np.arange(n_rois)-.25,
	showmeans = True, showmedians=True, widths=np.repeat(.25,n_rois))






