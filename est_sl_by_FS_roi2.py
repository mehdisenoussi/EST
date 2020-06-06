# for this script to work you need
# - nibabel (https://nipy.org/nibabel/)
#		conda install -c conda-forge nibabel 
import matplotlib as mpl
mpl.interactive(True)
from matplotlib import pyplot as pl

import numpy as np
import numpy.linalg as npl
import nibabel as nib
from nibabel.affines import apply_affine
from scipy import stats


# function compute the average voxel value (in our case classification accuracies) in a anatomically defined ROI
def get_avg_sl_acc_roi(sl_data, anat_rois, anat_vox2sl_vox, roi_inds):
	# - sl_data is a Nibabel Nifti image containing the searchlight results
	# - anat_rois is a Nibabel Nifti image containing the anatomical ROIs
	# automatically segmented using Freesurfer
	# - anat_vox2sl_vox is the transformation matrix for coordinates from
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
	roi_slspace_inds = apply_affine(anat_vox2sl_vox, roi_maskxyz).astype(np.int)

	# create a volume in the functional space to extract the values of the selected ROI
	roi_volmask_slspace = np.zeros(shape = sl_data.shape, dtype = np.bool)
	for roi_slspace_ind_n in roi_slspace_inds:
		roi_volmask_slspace[roi_slspace_ind_n[0], roi_slspace_ind_n[1], roi_slspace_ind_n[2]] = True

	cl_values = sl_data.get_data()[roi_volmask_slspace]
	cl_values = cl_values[cl_values > 0]
	# return the ROI mask indices in functional space, the ROI mask in volume in functional space and the average value of the selected ROI in the functional volume
	return roi_slspace_inds, roi_volmask_slspace, stats.kde.gaussian_kde(cl_values)



data_path = '/Volumes/mehdimac/ghent/side_projects/danesh/'
data_path_sl = '/Users/mehdi/work/ghent/side_projects/danesh/results/'
# data_path_sl = '/Volumes/mehdimac/ghent/side_projects/danesh/results/'

freesurfer_path = '/Applications/freesurfer/'


## We can load different atlases from the automatic FreeSurfer segementations,
## they differ in the (number of) areas that were segmented.
## these files are simply anatomical volumes in the native T1 space of the participant.
## They are coded as numbers, e.g. anterior cingulate will be voxels with a value of 34.
## you can look at all the parcels using the Freeview graphical interface included in
## FreeSurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI

# not so rich
# anat_rois = nib.load('/Users/mehdi/work/ghent/side_projects/danesh/data/Ouput-Filesscanner/Sub01/mri/aparc.DKTatlas+aseg.mgz')

# quite rich
# anat_rois = nib.load('/Users/mehdi/work/ghent/side_projects/danesh/data/Ouput-Filesscanner/Sub01/mri/aparc.a2009s+aseg.mgz')

# looks like the best
anat_rois = nib.load(data_path + 'data/Ouput-Filesscanner/Sub01/mri/aparc+aseg.mgz')

# get the codes of the segmented areas in the chosen segmentation volume
uniqs_rois_ind = np.unique(anat_rois.get_data())


# get the labels associated with each number.
# this table has the codes for all the segmentations so we need to find
# the labels in the file we loaded and find them in this table.
fs_labels = np.loadtxt(freesurfer_path + 'FreeSurferColorLUT.txt', dtype=np.str)
# create a dictionary to store all codes' labels
dict_code_label_all_seg = dict(zip(fs_labels[:, 0].astype(np.int), fs_labels[:, 1]))

# find the labels associated with the codes in our chosen segmentation volume
dict_code_label_this_seg = {}
for i in uniqs_rois_ind:
	dict_code_label_this_seg.update({i:fs_labels[fs_labels[:, 0] == np.str(i), 1]})


n_subj = 18


###### AREAS TO ADD ######
# parahippocampal, enthorinal, hippocampal
# check left right differences

## here are some examples ROIs just to show how to code them
## each element of rois_inds is the codes for an ROI, an element is a
## multi-element list all these ROIs will be concatenated (e.g. left and right parts of an ROI)
rois_inds = np.array([  [1025, 2025], # L&R precuneus cx
						[1025, 2025], # L&R entohirnal cx
						[1016, 2016], # L&R parahippocampal cx
						[17, 53],	  # L&R hippocampus
						[1026, 2026], # left and right rostral anterior cingulate cortex
						[1002, 2002], # left and right caudal anterior cingulate cortex
						[1003, 2003], # left and right caudal middle frontal cortex
						[1018, 2018], # left and right pars opercularis
						[1019, 2019], # left and right pars orbitalis
                        [1014, 2014], # medialorbitofrontal
                        [1012, 2012], # lateralorbitofrontal
                        [1032, 2032], # frontal pole
						[1011, 2011], # lateral occipital
						[1020, 2020]]) # left and right pars triangularis
n_rois = len(rois_inds)


classif_con = ['temporal_control_1', 'temporal_control_2',
				'temporal_context_1', 'subtask_context_1',
				'task_context_1'][-1]

# array to store the KDE object containing classification accuracies by ROI
rois_acc = np.empty(shape = [n_subj, n_rois], dtype=stats.kde.gaussian_kde)

########################################################################
# CHANGE THE NUMBER OF PERMUTATION HERE TO LOAD ALL COMPUTED SURROGATE DATA
########################################################################
n_perm = 1
########################################################################

rois_acc_perm = np.empty(shape = [n_subj, n_rois, n_perm], dtype=stats.kde.gaussian_kde)

for s_ind in range(n_subj):
	print('subj %i' % (s_ind+1))
	sl_data = nib.load(data_path_sl + 'grayMat/obs%02i_classif_%s_res_sl6_image.nii' % (s_ind, classif_con))
	anat_rois = nib.load(data_path + 'data/Ouput-Filesscanner/Sub%02i/mri/aparc+aseg.mgz' % (s_ind+1))
	
	# compute transformation of coordinates from anatomical to functional (searchlight volume)
	anat_vox2sl_vox = npl.inv(sl_data.affine).dot(anat_rois.affine)

	for roi_ind, roi_n_inds in enumerate(rois_inds):
		print('\troi %i' % (roi_ind+1))
		# for sub_roi_ind in roi_n_inds:
			# print('\t\troi: %s' % fs_labels[fs_labels[:, 0].astype(np.int) == sub_roi_ind, 1][0])
		roi_slspace_inds, roi_volmask_slspace, rois_acc[s_ind, roi_ind] =\
				get_avg_sl_acc_roi(sl_data, anat_rois, anat_vox2sl_vox, roi_n_inds)
	
	print('\tpermutations..')
	for perm_n in np.arange(n_perm):
		sl_data = nib.load(data_path_sl +\
			'grayMat/obs%02i_classif_%s_res_sl6_image_perm%i.nii'\
			% (s_ind, classif_con, perm_n))

		for roi_ind, roi_n_inds in enumerate(rois_inds):
			print('\troi %i' % (roi_ind+1))
			# for sub_roi_ind in roi_n_inds:
				# print('\t\troi: %s' % fs_labels[fs_labels[:, 0].astype(np.int) == sub_roi_ind, 1][0])
			roi_slspace_inds, roi_volmask_slspace, rois_acc_perm[s_ind, roi_ind, perm_n] =\
					get_avg_sl_acc_roi(sl_data, anat_rois, anat_vox2sl_vox, roi_n_inds)

	# plot this ROI in the functional space just to check its location
	# IF YOU UNCOMMENT THIS IT WILL PLOT ONE FIGURE PER ROI PER PARTICIPANT (which is a lot..)

		# sldata_copy = sl_data.get_data().copy()
		# sldata_copy[sldata_copy!=0] = 1
		# for roi_slspace_ind_n in roi_slspace_inds:
		# 	sldata_copy[roi_slspace_ind_n[0], roi_slspace_ind_n[1], roi_slspace_ind_n[2]] = 3

		# fig, axs = pl.subplots(5, 7)
		# for i in range(33):
		# 	axs.flatten()[i].imshow(sldata_copy[:, :, i], vmin=0, vmax=3)
		# 	axs.flatten()[i].set_xticklabels([])
		# 	axs.flatten()[i].set_yticklabels([])
		# axs[-1, -2].set_frame_on(False); axs[-1, -2].set_xticklabels([]); axs[-1, -2].set_yticklabels([]);
		# axs[-1, -1].set_frame_on(False); axs[-1, -1].set_xticklabels([]); axs[-1, -1].set_yticklabels([]);
		# roiname = fs_labels[fs_labels[:, 0].astype(np.int) == roi_n_inds[0], 1][0]
		# pl.suptitle('subj %i - roi %s' % (s_ind+1, roiname))

		# pl.savefig(\
		# 	'/Users/mehdi/work/ghent/side_projects/danesh/code/subj_roi_plots/roi_slices/subj%02i_%s.png' % (s_ind, roiname))
		# pl.close()

# failed attempt at plotting the selected areas in 3D just to check locations or ROIs
# couldn't install nipy, I'll get to it later
# nipy.labs.viz_tools.maps_3d.plot_map_3d(sldata_copy,
# 		sl_data.affine, anat=anat_rois.get_data(), anat_affine=anat_rois.affine)





###################################################################
######### 					AVERAGE 			###################
###################################################################

rois_acc_means = np.array([[rois_acc[s_ind, roi_ind].dataset.mean()\
	for roi_ind in np.arange(len(rois_inds))]\
		for s_ind in np.arange(n_subj)])

rois_acc_surr_means = np.zeros(shape = [n_subj, n_rois, n_perm])
for perm_n in np.arange(n_perm):
	rois_acc_surr_means[:, :, perm_n] =\
		np.array([[rois_acc_perm[s_ind, roi_ind, perm_n].dataset.mean()\
		for roi_ind in np.arange(len(rois_inds))]\
			for s_ind in np.arange(n_subj)])

plevels = np.array([.1,.05,.01,.005])
ptexts = np.array(['ms','*','**','***'])
font = {'family': 'arial', 'color':  'red',
        'weight': 'bold'}
fontsizes = np.array([10, 20, 20, 20])
## Plot the results per ROI
cols = []
pl.figure()
for roi_ind in np.arange(n_rois):
	temp = pl.errorbar(x = roi_ind+.33, y = rois_acc_means[:, roi_ind].mean(axis=0),
		yerr = rois_acc_means[:, roi_ind].std(axis=0)/np.sqrt(n_subj), fmt = 'o',
		label = dict_code_label_this_seg[rois_inds[roi_ind][0]][0])
	cols.append(temp.get_children()[0].get_c())
	t,p=stats.ttest_1samp(rois_acc_means[:, roi_ind], popmean=.5)
	if p<plevels[0]:
		pl.text(x=roi_ind, y=.4, s=ptexts[p<plevels][-1],
			fontsize=fontsizes[p<plevels][-1], fontdict=font)
# violins = pl.violinplot(rois_acc_means, positions = np.arange(n_rois), showmeans = True, showmedians=True)
violins = pl.violinplot(rois_acc_surr_means.mean(axis=0).T,
	positions = np.arange(n_rois), showmeans = True, showmedians=True)
# set the violin plots color to match the points
for roi_ind in np.arange(n_rois): violins['bodies'][roi_ind].set_color(cols[roi_ind])
pl.xlim(-1, n_rois); pl.ylim(.3, .7)
pl.legend(); pl.grid(); pl.hlines(y=.5, xmin=-1, xmax=n_rois)
pl.ylabel('Classification accuracy')
pl.suptitle('%s - average classif accuracy' % (classif_con))



###################################################################
######### 					zscore 				###################
###################################################################

zscore_rois_acc =\
		np.array([[(rois_acc_means[s_ind, roi_ind]\
			- rois_acc_surr_means[s_ind, roi_ind, :].mean())\
			/ rois_acc_surr_means[s_ind, roi_ind, :].std()
		for roi_ind in np.arange(len(rois_inds))]\
			for s_ind in np.arange(n_subj)])

plevels = np.array([.1,.05,.01,.005])
ptexts = np.array(['ms','*','**','***'])
font = {'family': 'arial', 'color':  'red',
        'weight': 'bold'}
fontsizes = np.array([10, 20, 20, 20])
## Plot the results per ROI
cols = []
pl.figure()
for roi_ind in np.arange(n_rois):
	temp = pl.errorbar(x = roi_ind+.33, y = zscore_rois_acc[:, roi_ind].mean(axis=0),
		yerr = zscore_rois_acc[:, roi_ind].std(axis=0)/np.sqrt(n_subj), fmt = 'o',
		label = dict_code_label_this_seg[rois_inds[roi_ind][0]][0])
	cols.append(temp.get_children()[0].get_c())
	t,p=stats.ttest_1samp(zscore_rois_acc[:, roi_ind], popmean=0)
	if p<plevels[0]:
		pl.text(x=roi_ind, y=.3, s=ptexts[p<plevels][-1],
			fontsize=fontsizes[p<plevels][-1], fontdict=font)
pl.xlim(-1, n_rois); pl.ylim(-2.5, 2.5)
pl.legend(); pl.grid(); pl.hlines(y=0, xmin=-1, xmax=n_rois)
pl.ylabel('Classification accuracy')
pl.suptitle('%s - average classif accuracy' % (classif_con))






###################################################################
######### 					SKEWNESS 			###################
###################################################################
rois_acc_skews = np.array([[stats.skew(rois_acc[s_ind, roi_ind].dataset, axis=1)[0]\
	for roi_ind in np.arange(len(rois_inds))]\
		for s_ind in np.arange(n_subj)])

## Plot the results per ROI
cols = []
pl.figure()
for roi_ind in np.arange(n_rois):
	temp = pl.errorbar(x = roi_ind+.33, y = rois_acc_skews[:, roi_ind].mean(axis=0),
		yerr = rois_acc_skews[:, roi_ind].std(axis=0)/np.sqrt(n_subj), fmt = 'o',
		label = dict_code_label_this_seg[rois_inds[roi_ind][0]][0])
	cols.append(temp.get_children()[0].get_c())
violins = pl.violinplot(rois_acc_skews, positions = np.arange(n_rois), showmeans = True, showmedians=True)
# set the violin plots color to match the points
for roi_ind in np.arange(n_rois): violins['bodies'][roi_ind].set_color(cols[roi_ind])
pl.ylim(-1, 1)
pl.legend()
pl.grid()
pl.hlines(y=0, xmin=-1, xmax=n_rois)
pl.xlim(-1, n_rois)
pl.ylabel('Skewness')
pl.suptitle('%s - skewness' % (classif_con))






###################################################################
######### 					KURTOSIS 			###################
###################################################################
rois_acc_kurts = np.array([[stats.kurtosis(rois_acc[s_ind, roi_ind].dataset, axis=1)[0]\
	for roi_ind in np.arange(len(rois_inds))]\
		for s_ind in np.arange(n_subj)])
## Plot the results per ROI
cols = []
pl.figure()
for roi_ind in np.arange(n_rois):
	temp = pl.errorbar(x = roi_ind+.33, y = rois_acc_kurts[:, roi_ind].mean(axis=0),
		yerr = rois_acc_kurts[:, roi_ind].std(axis=0)/np.sqrt(n_subj), fmt = 'o',
		label = dict_code_label_this_seg[rois_inds[roi_ind][0]][0])
	cols.append(temp.get_children()[0].get_c())
violins = pl.violinplot(rois_acc_kurts, positions = np.arange(n_rois), showmeans = True, showmedians=True)
# set the violin plots color to match the points
for roi_ind in np.arange(n_rois): violins['bodies'][roi_ind].set_color(cols[roi_ind])
pl.ylim(-1, 1)
pl.legend()
pl.grid()
pl.hlines(y=0, xmin=-1, xmax=n_rois)
pl.xlim(-1, n_rois)
pl.ylabel('Kurtosis')
pl.suptitle('%s - Kurtosis' % (classif_con))










###### PLOT FOR INDIVIDUAL SUBJECTS #######

xs = np.linspace(0, 1, 200)
for s_ind in np.arange(n_subj):
	fig, axs = pl.subplots(int(np.sqrt(n_rois))+1, int(np.sqrt(n_rois)))
	pl.suptitle('sub=%i' % s_ind, fontsize=8)
	for roi_ind in np.arange(n_rois):
		b = axs.flatten()[roi_ind].plot(xs, rois_acc[s_ind, roi_ind].evaluate(xs))
		areaname = dict_code_label_this_seg[rois_inds[roi_ind][0]][0].split('-')[-1]
		data = rois_acc[s_ind, roi_ind].dataset
		avg, sk, kur = data.mean(), stats.skew(data-.5, axis=1)[0], stats.kurtosis(data-.5, axis=1)
		axs.flatten()[roi_ind].set_title('%s\navg=%.3f - sk=%.3f - kur=%.3f' % \
			(areaname, avg, sk, kur), fontsize=8)
		axs.flatten()[roi_ind].grid()
		axs.flatten()[roi_ind].vlines(x=.5, ymin=axs.flatten()[roi_ind].get_ybound()[0], ymax=axs.flatten()[roi_ind].get_ybound()[1])

	pl.tight_layout()

	pl.savefig(data_path_sl + 'subj_roi_plots/2perms/subj%02i_roiKDE_%s.png' %\
		(s_ind, classif_con), dpi=120)
	pl.close()




