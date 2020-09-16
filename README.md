# Welcome!

This is a set of Matlab and Python scripts to perform the analyses for the study "Neural representations of task context and temporal order during action sequence execution" Shahnazian, Senoussi, Krebs, Verguts, Holroyd (submitted), preprint: [https://www.biorxiv.org/content/10.1101/2020.09.10.290965v1](https://www.biorxiv.org/content/10.1101/2020.09.10.290965v1)


These scripts were written and tested on Matlab 2015b and Python version 3.7. (see preprint for full list and versions of Python packages used).

The dataset used is available through this Open Science Framework repository of Holroyd et al. (2018): [https://osf.io/wxhta/](https://osf.io/wxhta/)


For simplicity and accessibility, we additionally uploaded the minimal data (GLM and Freesurfer outputs) from Holroyd et al. (2018) needed to reproduce our results on this OSF repository: [https://osf.io/4c9qu/?view_only=208f4e54692b440790e51f9f56e2b750](https://osf.io/4c9qu/?view_only=208f4e54692b440790e51f9f56e2b750).

To use these data to reproduce our results, unpack the zip file and change the paths in the scripts to the unpacked “data” directory.


To run any of the scripts you should be in the "EST" directory.

# Scripts order
To reproduce the results figures from the manuscript you need to run scripts in this order:
1. est_make_cortical_masks.py (Python)
2. save_info_from_SPM_file.m (Matlab)
3. est_save_beta_goodFormat.py (Python)
4. est_run_all_classifications.py (Python)
5. est_classif_stats_and_plot_ROI_analysis.py (Python, to reproduce Fig. 3)
6. est_classif_stats_and_plot_ROI_gener_analysis.py (Python, to reproduce Fig. 4)
7. est_classif_stats_and_plot_ROI_hemi_analysis.py (Python, to reproduce Fig. 5)

Let us know if you need any assistance to run these scripts using the contact information provided in our bioRxiv preprint (see above).
Thank you for your interest!
