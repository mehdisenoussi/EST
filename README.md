# Welcome!

This is a set of Matlab and Python scripts to perform the analyses for the study "Neural representations of task context and temporal order during action sequence execution" Shahnazian, Senoussi, Krebs, Verguts, Holroyd (submitted), preprint: [https://www.biorxiv.org/content/10.1101/2020.09.10.290965v1](https://www.biorxiv.org/content/10.1101/2020.09.10.290965v1)


These scripts were written and tested on Matlab 2015b and Python version 3.7. (see preprint for full list and versions of Python packages used)
The dataset used is available through this Open Science Framework repository of Holroyd et al. (2018): [https://osf.io/wxhta/](https://osf.io/wxhta/)

For simplicity and accessibility we will soon add another link to a repository with the data from Holroyd et al. (2018) and the data from Freesurfer we computed. [to come soon]

To run any of the scripts you should be in the "EST" directory and place the data from OSF in the "data" folder inside this directory.

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
