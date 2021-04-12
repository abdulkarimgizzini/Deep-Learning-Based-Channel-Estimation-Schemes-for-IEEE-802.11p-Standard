clc;clearvars;close all; warning('off','all');
% Load pre-defined DNN Testing Indices
load('DNN_TestingPacketsIndices.mat');
all_indices               = (1:1000).';
training_indices          = setdiff(all_indices,Testing_Packets_Indices);
Testing_Data_set_size     = size(Testing_Packets_Indices,1);
Training_Data_set_size    = 1000 - size(Testing_Packets_Indices,1);

% Define Simulation parameters
SNR                       = 0:5:30;
nUSC                      = 52;
nSym                      = 100;
mod                       = 'QPSK';
Ch                        = 'VTV_UC';
algo                      = 'DPA';

Training_DatasetX         = zeros(nUSC,nSym, Training_Data_set_size);
Training_DatasetY         = zeros(nUSC,nSym, Training_Data_set_size);
Testing_DatasetX          = zeros(nUSC,nSym, Testing_Data_set_size);
Testing_DatasetY          = zeros(nUSC,nSym, Testing_Data_set_size);

Train_X                   = zeros(nUSC*2, Training_Data_set_size * nSym);
Train_Y                   = zeros(nUSC*2, Training_Data_set_size * nSym);
Test_X                    = zeros(nUSC*2, Testing_Data_set_size * nSym);
Test_Y                    = zeros(nUSC*2, Testing_Data_set_size * nSym);



for n_snr = 1:size(SNR,2)
% Load simulation data according to the defined configurations (Ch, mod, algorithm) 
load(['D:\Simulation_',num2str(n_snr),'.mat'], 'True_Channels_Structure', [algo '_Structure']);
Algo_Channels_Structure = eval([algo '_Structure']);

% Select Training and Testing datasets
Training_DatasetX =  Algo_Channels_Structure(:,:,training_indices);
Training_DatasetY =  True_Channels_Structure(:,:,training_indices);
Testing_DatasetX =  Algo_Channels_Structure(:,:,Testing_Packets_Indices);
Testing_DatasetY =  True_Channels_Structure(: ,:,Testing_Packets_Indices);

% Expend Testing and Training Datasets
Training_DatasetX_expended = reshape(Training_DatasetX, nUSC, nSym * Training_Data_set_size);
Training_DatasetY_expended = reshape(Training_DatasetY, nUSC, nSym * Training_Data_set_size);
Testing_DatasetX_expended = reshape(Testing_DatasetX, nUSC, nSym * Testing_Data_set_size);
Testing_DatasetY_expended = reshape(Testing_DatasetY, nUSC, nSym * Testing_Data_set_size);

% Complex to Real domain conversion
Train_X(1:nUSC,:)           = real(Training_DatasetX_expended);
Train_X(nUSC+1:2*nUSC,:)    = imag(Training_DatasetX_expended);
Train_Y(1:nUSC,:)           = real(Training_DatasetY_expended);
Train_Y(nUSC+1:2*nUSC,:)    = imag(Training_DatasetY_expended);

Test_X(1:nUSC,:)              = real(Testing_DatasetX_expended);
Test_X(nUSC+1:2*nUSC,:)       = imag(Testing_DatasetX_expended);
Test_Y(1:nUSC,:)              = real(Testing_DatasetY_expended);
Test_Y(nUSC+1:2*nUSC,:)       = imag(Testing_DatasetY_expended);

% Save training and testing datasets to the DNN_Datasets structure
DNN_Datasets.('Train_X') =  Train_X;
DNN_Datasets.('Train_Y') =  Train_Y;
DNN_Datasets.('Test_X') =  Test_X;
DNN_Datasets.('Test_Y') =  Test_Y;

% Save the DNN_Datasets structure to the specified folder in order to be used later in the Python code 
save(['D:\' , algo, '_DNN_Dataset_' num2str(n_snr)],  'DNN_Datasets');
end