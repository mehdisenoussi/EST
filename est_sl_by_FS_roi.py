# for this script to work you need
# - nibabel (https://nipy.org/nibabel/)
#		conda install -c conda-forge nibabel 

import numpy as np
import numpy.linalg as npl
import nibabel as nib
from nibabel.affines import apply_affine
from matplotlib import pyplot as pl

# function compute the average voxel value (in our case classification accuracies) in a anatomically defined ROI
def get_avg_sl_acc_roi(sl_data, anat_rois, anat_vox2sl_vox, roi_inds):
	# sl_data is a Nibabel Nifti image containing the searchlight results
	# anat_rois is a Nibabel Nifti image containing the anatomical ROIs automatically segmented using Freesurfer
	# anat_vox2sl_vox is the transformation matrix for coordinates from anatomical to functional (searchlight)
	# roi_inds is a list of indices or codes referring to anatomical ROIs in the anat_rois volume

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
    for voxelnumber in range(len(roi_slspace_inds)): 
        print(str(voxelnumber))
        roi_volmask_slspace[roi_slspace_inds[voxelnumber,0], roi_slspace_inds[voxelnumber,1], roi_slspace_inds[voxelnumber,2]] = True

	# return the ROI mask indices in functional space, the ROI mask in volume in functional space and the average value of the selected ROI in the functional volume
    return roi_slspace_inds, roi_volmask_slspace, sl_data.get_data()[roi_volmask_slspace].mean()

data_path = 'C:\\Users\\danesh\\Desktop\\clayfmri\\'

freesurfer_path = 'C:\\Users\\danesh\\.conda\\envs\\mne\\Lib\\site-packages\\mne\\data\\'
subindices=[1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18]


## We can load different atlases from the automatic FreeSurfer segementations, they differ in the (number of) areas that were segmented.
## these files are simply anatomical volumes in the native T1 space of the participant.
## They are coded as numbers, e.g. anterior cingulate will be voxels with a value of 34.
## you can look at all the parcels using the Freeview graphical interface included in FreeSurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI
# not so rich
# anat_rois = nib.load('/Users/mehdi/work/ghent/side_projects/danesh/data/Ouput-Filesscanner/Sub01/mri/aparc.DKTatlas+aseg.mgz')

# quite rich
# anat_rois = nib.load('/Users/mehdi/work/ghent/side_projects/danesh/data/Ouput-Filesscanner/Sub01/mri/aparc.a2009s+aseg.mgz')

# looks like the best
anat_rois = nib.load(data_path + 'FS_outputs/Sub01/mri/aparc+aseg.mgz')


# get the codes of the segmented areas in the chosen segmentation volume
uniqs_rois_ind = np.unique(anat_rois.get_data())


# get the labels associated with each number.
# this table has the codes for all the segmentations so we need to find the labels in the file we loaded and find them in this table.
fs_labels = np.loadtxt(freesurfer_path + 'FreeSurferColorLUT.txt', dtype=np.str)
# create a dictionary to store all codes' labels
dict_code_label_all_seg = dict(zip(fs_labels[:, 0].astype(np.int), fs_labels[:, 1]))

# find the labels associated with the codes in our chosen segmentation volume
dict_code_label_this_seg = {} 
for i in uniqs_rois_ind:
    dict_code_label_this_seg.update({i:fs_labels[fs_labels[:, 0] == np.str(i), 1]})


n_subj = 18

## here are some examples ROIs just to show how to code them
## each element of rois_inds is the codes for an ROI, an element is a multi-element list all these ROIs will be concatenated (e.g. left and right parts of an ROI)
rois_inds = np.array([  #[1026, 2026], # left and right rostral anterior cingulate cortex
                        [1002, 2002], # left and right caudal anterior cingulate cortex
                        [1003, 2003], # left and right caudal middle frontal cortex
                        [1018, 2018]])#, # left and right pars opercularis
						# [1019, 2019], # left and right pars orbitalis
						# [1020, 2020]]) # left and right pars triangularis
n_rois = len(rois_inds)

# array to store the average classification accuracy by ROI
rois_acc = np.zeros(shape = [n_subj, n_rois])

for s_ind, s_id in enumerate(subindices):
    print('subj %i' % (s_ind+1))
....sl_data = nib.load(data_path + 'sl\\obs%02i_classif_temporal_context_1_res_sl6_image.nii' % s_id)
....anat_rois = nib.load(data_path + 'FS_outputs/Sub%02i/mri/aparc+aseg.mgz' % (s_id))
    anat_vox2sl_vox = npl.inv(sl_data.affine).dot(anat_rois.affine)
    for roi_index in len(rois_inds):
        roi_ind, roi_n_inds=rois_inds[:,roi_index]
............print('\t\troi: %s' % fs_labels[fs_labels[:, 0].astype(np.int) == sub_roi_ind, 1][0])
............roi_slspace_inds, roi_volmask_slspace, rois_acc[s_ind, roi_ind] = get_avg_sl_acc_roi(sl_data, anat_rois, anat_vox2sl_vox, roi_n_inds)

	
	## plot this ROI in the functional space just to check its location
	## IF YOU UNCOMMENT THIS IT WILL PLOT ONE FIGURE PER ROI PER PARTICIPANT (which is a lot..)

	# sldata_copy = sl_data.get_data().copy()
	# sldata_copy[sldata_copy!=0] = 1
	# for roi_slspace_ind_n in roi_slspace_inds:
	# 	sldata_copy[roi_slspace_ind_n[0], roi_slspace_ind_n[1], roi_slspace_ind_n[2]] = 3

	# pl.figure()
	# for i in range(33):
	# 	pl.subplot(5, 7, i+1)
	# 	pl.imshow(sldata_copy[:, :, i], vmin=0, vmax=3)
	# pl.suptitle('subj %i - roi %s' % (s_ind+1, fs_labels[fs_labels[:, 0].astype(np.int) == sub_roi_ind, 1][0]))

# failed attempt at plotting the selected areas in 3D just to check locations or ROIs
# couldn't install nipy, I'll get to it later
# nipy.labs.viz_tools.maps_3d.plot_map_3d(sldata_copy, sl_data.affine, anat=anat_rois.get_data(), anat_affine=anat_rois.affine)

## Plot the results per ROI
cols = []
pl.figure()
for roi_ind in np.arange(n_rois):
....temp = pl.errorbar(x = roi_ind+.33, y = rois_acc[:, roi_ind].mean(),
........yerr = rois_acc[:, roi_ind].std()/np.sqrt(n_subj), fmt = 'o',
........label = dict_code_label_this_seg[rois_inds[roi_ind][0]][0])
....cols.append(temp.get_children()[0].get_c())
violins = pl.violinplot(rois_acc, positions = np.arange(n_rois), showmeans = True, showmedians=True)
# set the violin plots color to match the points
for roi_ind in np.arange(n_rois): violins['bodies'][roi_ind].set_color(cols[roi_ind])
pl.ylim(.1, .9)
pl.legend()
pl.grid()
pl.hlines(y=.5, xmin=-1, xmax=n_rois)
pl.xlim(-1, n_rois)







