% script to extract each subject's Vbeta structre which contains information
% on each beta (e.g. coffee or tea sequence, etc.) from the SPM GLM files
%
% author: Mehdi Senoussi, 11/05/20

% change this path to the location where the preprocessed data are
data_path = '/Volumes/mehdimac/ghent/est/data/holroyd2018/';
% get all the directories there, they represent each subject
list = dir(data_path);
list = list(4:end);

% for each subject directory get the SPM.mat file which contains the variable Vbeta.
% Vbeta holds the information on which beta (e.g. beta103.nii) "represents what"
% e.g. beta103.nii is the action 3 (stirring) for a coffee making sequence with water first.
% we the save the Vbeta structure to be able to load it with Python in the searchlight analysis script
for li = list'
    load([data_path, li.name, '/GLM/SPM.mat'])
    Vbeta = SPM.Vbeta;
    % make sure to change this path
    save(sprintf('./results/Vbeta_%s.mat', li.name), 'Vbeta');
end
