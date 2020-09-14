from pathlib import Path
import numpy as np
from matplotlib import pyplot as pl
from scipy import stats
from statsmodels.stats import multitest as mt

res_path = Path('./results/')

n_obs = 18

analysis_to_run = 'roi_classif'

z = np.load(res_path / ('classif_res_%s.npz' % analysis_to_run), allow_pickle=True)['arr_0'][...,np.newaxis][0]
scores_all = z['scores_all']

# load ROIs info
z = np.load(res_path / ('rois_and_names_%s.npz' % analysis_to_run), allow_pickle=True)['arr_0'][...,np.newaxis][0]
rois_inds = z['rois_inds']
roi_names = z['roi_names']
n_rois = len(rois_inds)

n_contr = 2

n_draws = 32 * n_obs

font = {'family': 'arial', 'weight': 'bold'}
cols = ["#66b266", "#e74c3c"]

xs = np.arange(n_rois)
contr_inds = np.arange(n_contr)
x_shift = (contr_inds - n_contr/2)/n_contr/2

scores_allpc = scores_all.squeeze()*100
fig, ax = pl.subplots(1, 1)
ax.grid(zorder=0)
for cont_ind in np.arange(n_contr):
	sign_y = 67-cont_ind*2
	x_contr = xs + x_shift[cont_ind]
	ax.errorbar(x = x_contr,
		y=np.nanmean(np.nanmean(scores_allpc[..., cont_ind, :], axis=-1), axis=0),
		yerr=np.nanstd(np.nanmean(scores_allpc[..., cont_ind, :], axis=-1), axis=0)/(n_obs**.5),
		fmt='o', zorder=1, color=cols[cont_ind])

	toplot = np.nanmean(scores_allpc[..., cont_ind, :], axis=-1)

	ax.set_ylim([30, 70])
	ps = np.array([stats.binom_test(x=int((np.nanmean(scores_allpc[:, roi_ind, cont_ind, :])/100)*n_draws),
		n=n_draws, p=0.5, alternative='greater') for roi_ind in np.arange(n_rois)])
	sign, pfdr = mt.fdrcorrection(ps, alpha = 0.05)

	for roi_ind in np.arange(n_rois):
		p = pfdr[roi_ind]
		if p<.005: ax.text(x=x_contr[roi_ind], y=sign_y, s='***', fontdict=font, color=cols[cont_ind])
		elif p<.01: ax.text(x=x_contr[roi_ind], y=sign_y, s='**', fontdict=font, color=cols[cont_ind])
		elif p<.05: ax.text(x=x_contr[roi_ind], y=sign_y, s='*', fontdict=font, color=cols[cont_ind])

	ax.set_xticks(np.arange(n_rois))
	ax.set_xticklabels(roi_names, rotation=90)

ax.hlines(y=50, xmin=-1, xmax=n_rois+1, linestyle='--')
ax.set_xlim([-1, 10])
pl.ylabel('Classification accuracy (%)')
pl.title(analysis_to_run)
pl.subplots_adjust(left=.09, bottom=.4, right=.97, top=.85, wspace=.28, hspace=.32)




