clc;clearvars;close all; warning('off','all');
% Load pre-defined DNN Testing Indices
load('./samples_indices_100.mat');
configuration = 'testing'; % training or testing
% Define Simulation parameters
nUSC                      = 52;
nSym                      = 50;
mobility                  = 'High';
modu                      = 'QPSK';
ChType                    = 'VTV_SDWW';
scheme                    = 'STA';

if (isequal(configuration,'training'))
    indices = training_samples;
    EbN0dB           = 40; 
   
elseif(isequal(configuration,'testing'))
    indices = testing_samples;
    EbN0dB           = 0:5:40;    
end
Dataset_size     = size(indices,1);
Dataset_X        = zeros(nUSC*2, Dataset_size * nSym);
Dataset_Y        = zeros(nUSC*2, Dataset_size * nSym);

SNR                       = EbN0dB.';
N_SNR                     = length(SNR);
for n_snr = 1:N_SNR

load(['./',mobility,'_',ChType,'_',modu,'_',configuration,'_simulation_' num2str(EbN0dB(n_snr)),'.mat'], 'True_Channels_Structure', [scheme '_Structure']);
scheme_Channels_Structure = eval([scheme '_Structure']);
DatasetX_expended = reshape(scheme_Channels_Structure, nUSC, nSym * Dataset_size );
DatasetY_expended = reshape(True_Channels_Structure, nUSC, nSym * Dataset_size);

% Complex to Real domain conversion
Dataset_X(1:nUSC,:)           = real(DatasetX_expended); 
Dataset_X(nUSC+1:2*nUSC,:)    = imag(DatasetX_expended);
Dataset_Y(1:nUSC,:)           = real(DatasetY_expended);
Dataset_Y(nUSC+1:2*nUSC,:)    = imag(DatasetY_expended);

if (isequal(configuration,'training'))
    DNN_Datasets.('Train_X') =  Dataset_X.';
    DNN_Datasets.('Train_Y') =  Dataset_Y.';
elseif(isequal(configuration,'testing'))
    DNN_Datasets.('Test_X') =  Dataset_X.';
    DNN_Datasets.('Test_Y') =  Dataset_Y.';  
end
save(['./',mobility,'_',ChType,'_',modu,'_',scheme,'_DNN_',configuration,'_dataset_' num2str(EbN0dB(n_snr)),'.mat'],  'DNN_Datasets');

end
