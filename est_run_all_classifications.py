import glob
from pathlib import Path
import nibabel as nib
import numpy as np
# import classification library
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy.linalg as npl
from nibabel.affines import apply_affine
# import the cross-validation procedure
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_validate


# function compute the average voxel value (in our case classification accuracies) in a anatomically defined ROI
def get_anat_roi_func_coord(func_shape, anat_rois, anat_vox2func_vox, roi_inds, mask):
	# - func_shape is a tuple containing the shape of the functional data (i.e. betas)
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
	# transform these indices in the functional space and approximate their location in functional
	# space by turning them into integers
	roi_funcSpace_inds = apply_affine(anat_vox2func_vox, roi_maskxyz).astype(np.int)

	# create a volume in the functional space to extract the values of the selected ROI
	roi_volmask_funcSpace = np.zeros(shape = func_shape, dtype = np.bool)
	for roi_funcSpace_ind_n in roi_funcSpace_inds:
		roi_volmask_funcSpace[roi_funcSpace_ind_n[0], roi_funcSpace_ind_n[1], roi_funcSpace_ind_n[2]] = True

	roi_volmask_funcSpace = roi_volmask_funcSpace & mask

	# return the ROI mask indices in functional space, the ROI mask in volume in functional space
	return roi_funcSpace_inds, roi_volmask_funcSpace


################################################################
# make sure to change these paths
################################################################
# functional data paths
hd_data_path = Path('/Volumes/mehdimac/ghent/est/data/')
betas_data_path = hd_data_path / 'holroyd2018/'
anat_data_path = hd_data_path / 'freesurfer_output/'

# an installation of Freesurfer is needed to run the recon-all
# that yields the different ROIs and gray matter mask but also
# to get the ROI-to-code mapping (FreeSurferColorLUT.txt file)
#
# change this path according to where Freesufer is installed
# on your system
freesurfer_path = Path('/Applications/freesurfer/')

res_path = Path('./results/')
if not res_path.exists():
	res_path.mkdir()
################################################################

# get the name of each observer
obs_paths = glob.glob(betas_data_path.as_posix() + '/*')
obs_codes = np.array([op.split('/')[-1] for op in obs_paths])
n_obs = len(obs_codes)

# how many features to keep for feature selection
n_feat = 120

# One automatic FreeSurfer segementation
anat_rois = nib.load((anat_data_path / Path('Sub01/mri/aparc+aseg.mgz')).as_posix())
# get the codes of the segmented areas in the chosen segmentation volume
uniqs_rois_ind = np.unique(anat_rois.get_data())

# get the labels associated with each number.
fs_labels = np.loadtxt(freesurfer_path / 'FreeSurferColorLUT.txt', dtype=np.str)
# create a dictionary to store all codes' labels
dict_code_label_all_seg = dict(zip(fs_labels[:, 0].astype(np.int), fs_labels[:, 1]))

# find the labels associated with the codes in our chosen segmentation volume
dict_code_label_this_seg = {}
for i in uniqs_rois_ind:
	dict_code_label_this_seg.update({i:fs_labels[fs_labels[:, 0] == np.str(i), 1]})


################################################################################################
################################################################################################
#																							   #
#										Which analysis to run 								   #
#																							   #
################################################################################################
################################################################################################

analysis_to_run = 'roi_classif'
# analysis_to_run = 'roi_cross-classif'
# analysis_to_run = 'roi_classif_hemi'

################################################################################################
################################################################################################


if analysis_to_run in ['roi_classif', 'roi_cross-classif']:
	# ROIs
	rois_inds = np.array([  \
							[1016, 2016], # L&R parahippocampal cx
							[17, 53],	  # L&R hippocampus
							[1015, 2015], # L&R Middle Temporal
							[1002, 2002], # L&R caudal anterior cingulate cortex
							[1003, 2003], # L&R caudal middle frontal cortex
							[1027, 2027], # L&R rostral middle frontal
							[1018, 2018], # opercularis
							[1020, 2020], # L&R pars triangularis
							[1019, 2019], # L&R pars orbitalis
							[1032, 2032], # L&R frontal pole
						])

elif analysis_to_run=='roi_classif_hemi':
	# ROI per hemisphere
	rois_inds = np.array([  \
							[1016], [2016], # L&R parahippocampal cx
							[17], [ 53],	  # L&R hippocampus
							[1015], [ 2015], # L&R Middle Temporal
							[1002], [ 2002], # L&R caudal anterior cingulate cortex
							[1003], [ 2003], # L&R caudal middle frontal cortex
							[1027], [ 2027], # L&R rostral middle frontal
							[1018], [ 2018], # opercularis
							[1020], [ 2020], # L&R pars triangularis\
							[1019], [ 2019], # L&R pars orbitalis	
						])

n_rois = len(rois_inds)

if analysis_to_run in ['roi_classif', 'roi_classif_hemi']:
	classif_cons = ['temporal_contrast', 'context_contrast']
	n_fold = 4

elif analysis_to_run == 'roi_cross-classif':
	classif_cons = ['cross_temporal_contrast', 'cross_context_contrast']
	n_fold = 2

n_contr = len(classif_cons)
scores_all = np.zeros(shape = [n_obs, n_rois, n_contr, n_fold])
train_scores_all = np.zeros(shape = [n_obs, n_rois, n_contr, n_fold])

for obs_ind in np.arange(n_obs):
	print('obs %i' % obs_ind)
	print('\tload data...')
	obs_num = obs_ind + 1
	# load betas and labels
	z = np.load((res_path / ('betas_and_affine_obs%02i.npz' %\
			int(obs_codes[obs_ind].split('_')[-1]))).as_posix(), allow_pickle=True)
	z = z['arr_0'][..., np.newaxis][0]
	beta_of_int = z['beta_of_int']
	coffee_or_tea = z['coffee_or_tea']
	water_order = z['water_order']
	action_n = z['action_n']
	block = z['block']
	func_affine = z['func_affine']
	func_shape = z['func_shape']


	anat_rois = nib.load((anat_data_path /\
		('Sub%02i/mri/aparc+aseg.mgz' % obs_num)).as_posix())
	# gray matter mask
	grayMat_mask = nib.load((res_path /\
		('masks/obs%02i_ribbon_grayMatter.nii' %\
				obs_num)).as_posix()).get_data().astype(np.bool)

	# compute transformation of coordinates from anatomical to functional
	anat_vox2func_vox = npl.inv(func_affine).dot(anat_rois.affine)

	for roi_ind, roi_n_inds in enumerate(rois_inds):
		print('\troi %i' % (roi_ind+1))
		roi_funcSpace_inds, roi_volmask_funcSpace =\
				get_anat_roi_func_coord(func_shape, anat_rois,
					anat_vox2func_vox, roi_n_inds, grayMat_mask)

		# check if the number of voxels is higher than n_feat in this ROI
		if roi_volmask_funcSpace.sum() < n_feat:
			print('\t\t!!!!!\t\tNOT ENOUGH VOXELS IN THIS ROI (%i voxels)\t\t!!!!!\n\n' %\
				roi_volmask_funcSpace.sum())
			n_feat_to_use = roi_volmask_funcSpace.sum()
		else:
			n_feat_to_use = n_feat

		betas_roi_n = beta_of_int[:, roi_volmask_funcSpace]

		### the different contrasts
		for contr_ind, classif_con in enumerate(classif_cons):
			print('\t\tcontrast: %s' % classif_con)
			if classif_con == 'temporal_contrast':
				first_or_second_stir = (action_n == 3) | (action_n == 5)
				mask_classif = first_or_second_stir
				y = (action_n[mask_classif] == np.unique(action_n[mask_classif])[0]).astype(np.int)
				groups = None
				cv = KFold(n_splits=n_fold)
				
			elif classif_con == 'context_contrast':
				coffee_or_tea_stirs = ((coffee_or_tea == 'c') & ((action_n == 3) | (action_n == 5))) |\
									  ((coffee_or_tea == 't') & ((action_n == 3) | (action_n == 5)))
				mask_classif = coffee_or_tea_stirs
				y = (coffee_or_tea[mask_classif] == 't').astype(np.int)
				groups = None
				cv = KFold(n_splits=n_fold)
				
			elif classif_con == 'cross_temporal_contrast':
				first_or_second_stir = (action_n == 3) | (action_n == 5)
				mask_classif = first_or_second_stir
				y = (action_n[mask_classif] == np.unique(action_n[mask_classif])[0]).astype(np.int)
				groups = (coffee_or_tea[mask_classif] == 't').astype(np.int)
				cv = LeaveOneGroupOut()

			elif classif_con == 'cross_context_contrast':
				coffee_stirs = ((coffee_or_tea == 'c') & ((action_n == 3) | (action_n == 5)))
				tea_stirs = ((coffee_or_tea == 't') & ((action_n == 3) | (action_n == 5)))
				mask_classif = coffee_stirs | tea_stirs
				y = (coffee_or_tea[mask_classif] == 't').astype(np.int)
				groups = (block[mask_classif] > 2).astype(np.int)
				cv = LeaveOneGroupOut()

			# normalize beta values per sample	
			beta_to_classif = betas_roi_n[mask_classif, :]
			beta_to_classif = (beta_to_classif - beta_to_classif.mean(axis=-1)[:, np.newaxis])\
				/ beta_to_classif.std(axis=-1)[:, np.newaxis]

			# anova filter, take n_feat best ranked features
			anova_filter = SelectKBest(f_regression, k=n_feat_to_use)
			anova_svm = Pipeline([('scaler', StandardScaler()),
								('anova', anova_filter),
								('clf', svm.SVC(kernel = 'linear', max_iter = 3000))])

			temp_classif = cross_validate(anova_svm, beta_to_classif, y, groups=groups,
				cv = cv, n_jobs = -1, scoring='accuracy', return_train_score=True)
			
			scores_all[obs_ind, roi_ind, contr_ind, :] = temp_classif['test_score']
			train_scores_all[obs_ind, roi_ind, contr_ind, :] = temp_classif['train_score']


np.savez(res_path / ('classif_res_%s.npz' % analysis_to_run),
	{'scores_all':scores_all, 'train_scores_all':train_scores_all})


roi_names = [dict_code_label_this_seg[rois_inds[roi_ind][0]][0].split('-')[-1] for roi_ind in np.arange(n_rois)]

np.savez(res_path / ('rois_and_names_%s.npz' % analysis_to_run), {'roi_names':roi_names, 'rois_inds':rois_inds})



