from nilearn.image import new_img_like, load_img, smooth_img
import glob, time
import nibabel as nib
import numpy as np

res_path = '/Users/mehdi/work/ghent/side_projects/danesh/results/'


glm_data_path = '/Volumes/MEHDIFAT/misc/clayfmri/neuralData/'
obs_paths = glob.glob(glm_data_path + '*')
# get the name of each observer
obs_codes = np.array([op.split('/')[-1] for op in obs_paths])

for s_ind in np.arange(18):
	print('subj %i' % s_ind)
	# load the brain mask
	# anat_mask = nib.load(data_path + 'data/Ouput-Filesscanner/Sub%02i/mri/brainmask.mgz' % (s_ind+1))
	anat_mask = nib.load(data_path + 'data/Ouput-Filesscanner/Sub%02i/mri/ribbon.mgz' % (s_ind+1))

	glm_mask = nib.load(glm_data_path + obs_codes[s_ind] + '/GLM/mask.nii')

	# compute transformation of coordinates from anatomical to functional (searchlight volume)
	anat_vox2sl_vox = npl.inv(glm_mask.affine).dot(anat_mask.affine)


	############# brain mask #############

	# select only gray and white matter in each hemisphere
	anat_mask_data = (anat_mask.get_data().copy() == 2).astype(np.bool) |\
		(anat_mask.get_data().copy() == 3).astype(np.bool) |\
		(anat_mask.get_data().copy() == 41).astype(np.bool) |\
		(anat_mask.get_data().copy() == 42).astype(np.bool)

	x, y, z = np.where(anat_mask_data)
	mask_maskxyz = np.vstack([x, y, z]).T
	# transform these indices in the functional space and approximate their location
	# in functional space by turning them into integers
	mask_slspace_inds = apply_affine(anat_vox2sl_vox, mask_maskxyz).astype(np.int)

	# create a volume in the functional space to extract the values of the selected ROI
	mask_volmask_slspace = np.zeros(shape = glm_mask.get_data().shape, dtype = np.bool)
	for mask_slspace_ind_n in mask_slspace_inds:
		mask_volmask_slspace[mask_slspace_ind_n[0], mask_slspace_ind_n[1], mask_slspace_ind_n[2]] = True

	mask_anat_fs_final = (glm_mask.get_data()>0) & mask_volmask_slspace

	res_img = new_img_like(glm_mask, data=mask_anat_fs_final.astype(np.float))
	res_img.to_filename(res_path + 'masks/obs%02i_ribbon_grayAndWhiteMatter.nii' % (s_ind))



	############# gray matter mask #############

	# select only gray matter in each hemisphere
	anat_mask_data = (anat_mask.get_data().copy() == 3).astype(np.bool) |\
		(anat_mask.get_data().copy() == 42).astype(np.bool)

	x, y, z = np.where(anat_mask_data)
	mask_maskxyz = np.vstack([x, y, z]).T
	mask_slspace_inds = apply_affine(anat_vox2sl_vox, mask_maskxyz).astype(np.int)

	mask_volmask_slspace = np.zeros(shape = glm_mask.get_data().shape, dtype = np.bool)
	for mask_slspace_ind_n in mask_slspace_inds:
		mask_volmask_slspace[mask_slspace_ind_n[0], mask_slspace_ind_n[1], mask_slspace_ind_n[2]] = True

	mask_anat_fs_final = (glm_mask.get_data()>0) & mask_volmask_slspace

	res_img = new_img_like(glm_mask, data=mask_anat_fs_final.astype(np.float))
	res_img.to_filename(res_path + 'masks/obs%02i_ribbon_grayMatter.nii' % (s_ind))



	# mask_volmask_slspace = smooth_img(res_img, .8).get_data().astype(np.bool)

	# load subject's functional data
	one_beta_filename = glm_data_path + obs_codes[s_ind] + '/GLM/beta_0%03i.nii' % (1)
	one_beta_data = nib.load(one_beta_filename)


	fig, axs = pl.subplots(5, 7)
	for i in range(33):
		glm_data_toplot = np.logical_not(np.isnan(one_beta_data.get_data()[:,:,i])).astype(np.int)
		axs.flatten()[i].imshow(glm_data_toplot,vmin=0, vmax=3, alpha = .5)
		axs.flatten()[i].imshow((mask_anat_fs_final[:,:,i]>0).astype(np.int)+2, vmin=0, vmax=3, alpha = .5)
		axs.flatten()[i].set_xticklabels([])
		axs.flatten()[i].set_yticklabels([])
	axs[-1, -2].set_frame_on(False); axs[-1, -2].set_xticklabels([]); axs[-1, -2].set_yticklabels([]);
	axs[-1, -1].set_frame_on(False); axs[-1, -1].set_xticklabels([]); axs[-1, -1].set_yticklabels([]);
	pl.savefig(res_path + 'masks/mask_slices/subj%02i_graymatter.png' % s_ind)
	pl.close()















