clc;clearvars;close all; warning('off','all');
% Loading Simulation Data
Ch = 'VTV_UC';
mod = 'QPSK';
scheme = 'DPA';
DNN_Hidden_Layers ='402040';
Mod_Type                  = 1;              % 0 for BPSK and 1 for QAM 
if(Mod_Type == 0)
    nBitPerSym            = 1;
    Pow                   = 1;
    %BPSK Modulation Objects
    bpskModulator         = comm.BPSKModulator;
    bpskDemodulator       = comm.BPSKDemodulator;
    M                     = 1;
elseif(Mod_Type == 1)
    if(strcmp(mod,'QPSK') == 1)
         nBitPerSym            = 2; 
    elseif (strcmp(mod,'16QAM') == 1)
         nBitPerSym            = 4; 
    elseif (strcmp(mod,'64QAM') == 1)
         nBitPerSym            = 6; 
    end
    %nBitPerSym            = 2; % Number of bits per Data Subcarrier (2 --> 4QAM, 4 --> 16QAM, 6 --> 64QAM)
    M                     = 2 ^ nBitPerSym; % QAM Modulation Order   
    Pow                   = mean(abs(qammod(0:(M-1),M)).^2); % Normalization factor for QAM        
end

load(['./Simulations/',Ch,'_',mod,'_','Simulation_variables']);
load('./Datasets/Training_Testing_Indices');
N_Test_Frames = Testing_Data_set_size;
EbN0dB                    = (0:5:30)';
M                         = 2 ^ nBitPerSym;
Pow                       = mean(abs(qammod(0:(M-1),M)).^2); 
nSym                      = 50;
constlen                  = 7;
trellis                   = poly2trellis(constlen,[171 133]);
tbl                       = 34;
scramInit                 = 93;
nDSC                      = 48;
nUSC                      = 52;
Interleaver_Rows          = 16;
Interleaver_Columns       = (nBitPerSym * nDSC * nSym) / Interleaver_Rows;

N_SNR                      = size(EbN0dB,1);
Phf                        = zeros(N_SNR,1);
 
Err_DNN                 = zeros(N_SNR,1);
Ber_DNN                 = zeros(N_SNR,1);

dpositions             = [1:6, 8:20, 22:31, 33:45, 47:52].'; 
for i = 1:N_SNR 
    
    % Loading Simulation Parameters Results
    load(['./Simulations/',Ch,'_',mod,'_','Simulation_',num2str(i),'.mat']);
      
     % Loading DNN Results
     load(['./Results/',Ch,'_',mod,'_',scheme,'_DNN_',DNN_Hidden_Layers,'_Results_' num2str(i),'.mat']);
     
     Testing_X = eval([scheme,'_DNN_',DNN_Hidden_Layers,'_test_x_',num2str(i)]);
     Testing_X = reshape(Testing_X(1:52,:) + 1i*Testing_X(53:104,:), nUSC, nSym, N_Test_Frames);
     Testing_Y = eval([scheme,'_DNN_',DNN_Hidden_Layers,'_test_y_',num2str(i)]);
     Testing_Y = reshape(Testing_Y(1:52,:) + 1i*Testing_Y(53:104,:), nUSC, nSym, N_Test_Frames);
     DNN_Y = eval([scheme,'_DNN_',DNN_Hidden_Layers,'_corrected_y_',num2str(i)]);
     DNN_Y = reshape(DNN_Y(1:52,:) + 1i*DNN_Y(53:104,:), nUSC, nSym, N_Test_Frames);

    disp(['Running Simulation, SNR = ', num2str(EbN0dB(i))]);
    tic;
    for u = 1:N_Test_Frames
         c = Testing_Packets_Indices(1,u);        
         Phf(i)  = Phf(i)  + mean(sum(abs(True_Channels_Structure(:,:,c)).^2));
        
        % DNN 
        H_DNN = DNN_Y(:,:,u);
        Err_DNN (i) =  Err_DNN (i) +  mean(sum(abs(H_DNN - True_Channels_Structure(:,:,c)).^2)); 
        Bits_DNN  = de2bi(qamdemod(sqrt(Pow) * (Received_Symbols_FFT_Structure(dpositions ,:,c) ./ H_DNN(dpositions,:)),M));
        Ber_DNN (i) = Ber_DNN (i) + biterr(wlanScramble((vitdec((matintrlv((deintrlv(Bits_DNN(:),Random_permutation_Vector)).',Interleaver_Columns,Interleaver_Rows).'),trellis,tbl,'trunc','hard')),scramInit),TX_Bits_Stream_Structure(:,c));
      
        
    end
   toc;
end
Phf = Phf ./ N_Test_Frames;

ERR_DNN = Err_DNN ./ (N_Test_Frames * Phf); 
BER_DNN = Ber_DNN ./ (N_Test_Frames * nSym * nDSC * nBitPerSym);


%% Load NMSE and BER Results for Plotting

load(['./Results/',Ch,'_',mod,'_','BER_NMSE'],'BER_Ideal','BER_LS','BER_STA','BER_CDP','BER_TRFI','BER_MMSE_VP','BER_Initial',...
    'ERR_LS','ERR_STA','ERR_CDP','ERR_TRFI','ERR_MMSE_VP','ERR_Initial');

figure,
p1 = semilogy(EbN0dB, BER_Ideal,'k-o','LineWidth',2);
hold on;
p2 = semilogy(EbN0dB, BER_LS,'k--o','LineWidth',2);
hold on;
p3 = semilogy(EbN0dB, BER_STA,'g-o','LineWidth',2);
hold on;
p4 = semilogy(EbN0dB, BER_Initial,'r-^','LineWidth',2);
hold on;
p5 = semilogy(EbN0dB, BER_CDP,'b-+','LineWidth',2);
hold on;
p6 = semilogy(EbN0dB, BER_TRFI,'c-d','LineWidth',2);
hold on;
p7 = semilogy(EbN0dB, BER_MMSE_VP,'m-v','LineWidth',2);
hold on;
p8 = semilogy(EbN0dB, BER_DNN,'r--d','LineWidth',2);
hold on;
grid on;
legend([p1(1),p2(1),p3(1),p4(1),p5(1),p6(1),p7(1),p8(1)],{'Perfect Channel','LS','STA','Initial Channel','CDP','TRFI','MMSE-VP','DPA-DNN'});
xlabel('SNR(dB)');
ylabel('Bit Error Rate (BER)');

figure,
p1 = semilogy(EbN0dB,ERR_LS,'k--o','LineWidth',2);
hold on;
p2 = semilogy(EbN0dB,ERR_STA,'g-o','LineWidth',2);
hold on;
p3 = semilogy(EbN0dB,ERR_Initial,'r-^','LineWidth',2);
hold on;
p4 = semilogy(EbN0dB,ERR_CDP,'b-+','LineWidth',2);
hold on;
p5 = semilogy(EbN0dB,ERR_TRFI,'c-d','LineWidth',2);
hold on;
p6 = semilogy(EbN0dB,ERR_MMSE_VP,'m-v','LineWidth',2);
hold on;
p7 = semilogy(EbN0dB, ERR_DNN,'r--d','LineWidth',2);
hold on;
grid on;
legend([p1(1),p2(1),p3(1),p4(1),p5(1),p6(1),p7(1)],{'LS','STA','Initial Channel','CDP','TRFI','MMSE-VP','DPA-DNN'});
xlabel('SNR(dB)');
ylabel('Normalized Mean Sqaure Error (NMSE)');




