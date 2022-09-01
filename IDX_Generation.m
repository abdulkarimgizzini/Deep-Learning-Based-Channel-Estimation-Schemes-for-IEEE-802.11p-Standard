clc;clearvars; close all;
IDX = 100; % total number of samples
TR  = 0.80; % training dataset percentage
TE  = 0.20; % testing dataset percentage
All_IDX = 1:IDX;
training_samples = randperm(IDX,TR*IDX).'; % choose randomly 80% of the total indices for training
testing_samples = setdiff(All_IDX,training_samples).';
save(['./samples_indices_',num2str(IDX),'.mat'],  'training_samples','testing_samples'); 
