# script to plot and compute stats on the ROI classif analysis per HEMISPHERE
#
# author: Mehdi Senoussi, 24/08/20

from pathlib import Path
import numpy as np
from matplotlib import pyplot as pl
from scipy import stats
from statsmodels.stats import multitest as mt
import pandas as pd

res_path = Path('./results/')

n_obs = 18

analysis_to_run = 'roi_classif_hemi'

z = np.load(res_path / ('classif_res_%s.npz' % analysis_to_run), allow_pickle = True)['arr_0'][...,np.newaxis][0]
scores_all = z['scores_all']

# load ROIs info
z = np.load(res_path / ('rois_and_names_%s.npz' % analysis_to_run), allow_pickle = True)['arr_0'][...,np.newaxis][0]
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
		fmt='.', zorder=1, color=cols[cont_ind])
	for roi_ind in np.arange(n_rois):
		ax.plot(x_contr[roi_ind],
			np.nanmean(np.nanmean(scores_allpc[:, roi_ind, cont_ind, :], axis=-1), axis=0),
			['<', '>'][roi_ind%2], ms=8, zorder=2, color=cols[cont_ind])

	toplot = np.nanmean(scores_allpc[..., cont_ind, :], axis=-1)

	ax.set_ylim([30, 70])
	ax.set_xticks(np.arange(n_rois))
	ax.set_xticklabels(roi_names, rotation=90)

ax.hlines(y=50, xmin=-1, xmax=n_rois+1, linestyle='--')
ax.set_xlim([-1, 20])
pl.ylabel('Classification accuracy (%)')
pl.title(analysis_to_run)
pl.subplots_adjust(left=.09, bottom=.4, right=.97, top=.85, wspace=.28, hspace=.32)















### ANOVA
scores_forAnova = scores_allpc.mean(axis=-1)

roi_names = np.array(roi_names)

df_arr_cl_scores = np.zeros(shape = [n_obs*n_contr*n_rois, 5])#, dtype='<U40')
index = 0
for obs_ind in np.arange(n_obs):
	for contr_ind in np.arange(n_contr):
		for roi_ind in np.arange(0, n_rois, 2):
			for hemi_ind in np.arange(2):
				df_arr_cl_scores[index, :] = np.array([obs_ind,
					contr_ind, roi_ind, hemi_ind, scores_forAnova[obs_ind, roi_ind+hemi_ind, contr_ind]])
				index += 1
df_scores = pd.DataFrame(df_arr_cl_scores, columns=['obs', 'contrast', 'ROI', 'hemi', 'score'])


from statsmodels.stats.anova import AnovaRM
aovrm = AnovaRM(data=df_scores, depvar='score', subject='obs', within=['contrast', 'ROI', 'hemi'], aggregate_func=np.nanmean)
res = aovrm.fit()
print(res.summary())


######## POST-HOC Tests ########

# hemisphere by contrast (irrespective of ROIs)
t, p = stats.ttest_rel(scores_forAnova[:, np.arange(0, 18, 2), 0].mean(axis=1),
	scores_forAnova[:, np.arange(0, 18, 2), 1].mean(axis=1), axis=0)
print('Difference between Temporal and Context contrasts in Left hemisphere: t(17)=%.2f, p=%.5f'%(t, p))

t, p = stats.ttest_rel(scores_forAnova[:, np.arange(1, 18, 2), 0].mean(axis=1),
	scores_forAnova[:, np.arange(1, 18, 2), 1].mean(axis=1), axis=0)
print('Difference between Temporal and Context contrasts in Right hemisphere: t(17)=%.2f, p=%.5f'%(t, p))


# difference between hemisphere "intra-contrast"
# (only for ROIs showing sign. classif accuracy in ROI classif analysis)
# for tempral classif first
print('Temporal contrast')
ts = []; ps = []
for roi_ind in np.arange(0, n_rois, 2):
	if roi_ind in [0, 4, 8, 10, 12, 14, 16]:
		t, p = stats.ttest_rel(scores_forAnova[:, roi_ind, 0],
			scores_forAnova[:, roi_ind+1, 0], axis=0)
		print('\tROI: %s, Left vs Right: t(17)=%.2f, p=%.5f'%\
			(roi_names[roi_ind], t, p))
		ts.append(t); ps.append(p)

print('\n\nContext contrast')
for roi_ind in np.arange(0, n_rois, 2):
	if roi_ind in [14, 16]:
		t, p = stats.ttest_rel(scores_forAnova[:, roi_ind, 1],
			scores_forAnova[:, roi_ind+1, 1], axis=0)
		print('\tROI: %s, Left vs Right: t(17)=%.2f, p=%.5f'%\
			(roi_names[roi_ind], t, p))
		ts.append(t); ps.append(p)





