from nilearn.image import new_img_like
import glob, time
import nibabel as nib
import numpy as np
import numpy.linalg as npl
from nibabel.affines import apply_affine


res_path = Path('./results/')
if not res_path.exists():
	res_path.mkdir()

save_mask_path = res_path / 'masks/'
if not save_mask_path.exists():
	save_mask_path.mkdir()

hd_data_path = Path('/Volumes/mehdimac/ghent/est/data/')
betas_data_path = hd_data_path / 'holroyd2018/'
anat_data_path = hd_data_path / 'freesurfer_output/'

obs_paths = glob.glob(betas_data_path.as_posix() + '/*')
# get the name of each observer
obs_codes = np.array([op.split('/')[-1] for op in obs_paths])

for s_ind in np.arange(18):
	print('subj %i' % s_ind)
	# load the brain mask
	anat_mask = nib.load((anat_data_path / Path('Sub%02i/mri/ribbon.mgz' % (s_ind+1))).as_posix())

	glm_mask = nib.load((betas_data_path / Path('%s/GLM/mask.nii' % obs_codes[s_ind])).as_posix())

	# compute transformation of coordinates from anatomical to functional (searchlight volume)
	anat_vox2func_vox = npl.inv(glm_mask.affine).dot(anat_mask.affine)


	############# gray matter mask #############

	# select only gray matter in each hemisphere
	anat_mask_data = (anat_mask.get_data().copy() == 3).astype(np.bool) |\
		(anat_mask.get_data().copy() == 42).astype(np.bool)

	x, y, z = np.where(anat_mask_data)
	mask_maskxyz = np.vstack([x, y, z]).T
	mask_funcSpace_inds = apply_affine(anat_vox2func_vox, mask_maskxyz).astype(np.int)

	mask_volmask_funcSpace = np.zeros(shape = glm_mask.get_data().shape, dtype = np.bool)
	for mask_funcSpace_ind_n in mask_funcSpace_inds:
		mask_volmask_funcSpace[mask_funcSpace_ind_n[0], mask_funcSpace_ind_n[1], mask_funcSpace_ind_n[2]] = True

	mask_anat_fs_final = (glm_mask.get_data()>0) & mask_volmask_funcSpace

	res_img = new_img_like(glm_mask, data=mask_anat_fs_final.astype(np.float))
	res_img.to_filename((save_mask_path / ('obs%02i_ribbon_grayMatter.nii' % (s_ind+1))).as_posix())


