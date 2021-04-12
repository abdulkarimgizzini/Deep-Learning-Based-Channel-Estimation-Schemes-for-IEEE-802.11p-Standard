clc;clearvars;close all; warning('off','all');
% Loading Simulation Data
Ch = 'VTV_UC';
mod = 'QPSK';
nBitPerSym  = 2;
load(['D:\Simulation_variables.mat']);
load('DNN_TestingPacketsIndices');
N_Test_Frames = size(Testing_Packets_Indices,1);
EbN0dB                    = (0:5:30)';
M                         = 2 ^ nBitPerSym;
Pow                       = mean(abs(qammod(0:(M-1),M)).^2); 
nSym                      = 100;
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
 
Err_AE_DNN                 = zeros(N_SNR,1);
Ber_AE_DNN                 = zeros(N_SNR,1);

Err_STA_DNN1               = zeros(N_SNR,1);
Ber_STA_DNN1               = zeros(N_SNR,1);

Err_STA_DNN2               = zeros(N_SNR,1);
Ber_STA_DNN2               = zeros(N_SNR,1);

Err_TRFI_DNN               = zeros(N_SNR,1);
Ber_TRFI_DNN               = zeros(N_SNR,1);
dpositions             = [1:6, 8:20, 22:31, 33:45, 47:52].'; 
for i = 1:N_SNR 
    
    % Loading Simulation Parameters Results
    load(['D:\Simulation_',num2str(i),'.mat']);
      
     % Loading AE-DNN Results
     load(['D:\DPA_DNN_402040_Results_' num2str(i),'.mat']);
     DPA_Testing_X = eval(['DPA_DNN_402040_test_x_',num2str(i)]);
     DPA_Testing_X = reshape(DPA_Testing_X(1:52,:) + 1i*DPA_Testing_X(53:104,:), nUSC, nSym, N_Test_Frames);
     DPA_Testing_Y = eval(['DPA_DNN_402040_test_y_',num2str(i)]);
     DPA_Testing_Y = reshape(DPA_Testing_Y(1:52,:) + 1i*DPA_Testing_Y(53:104,:), nUSC, nSym, N_Test_Frames);
     DPA_DNN_Y = eval(['DPA_DNN_402040_corrected_y_',num2str(i)]);
     DPA_DNN_Y = reshape(DPA_DNN_Y(1:52,:) + 1i*DPA_DNN_Y(53:104,:), nUSC, nSym, N_Test_Frames);

     
     % Loading STA-DNN1 (15-10-15) and STA_DNN2(15-15-15) Results
     load(['D:\STA_DNN_151015_Results_' num2str(i),'.mat']);
     STA1_Testing_X = eval(['STA_DNN_151015_test_x_',num2str(i)]);
     STA1_Testing_X = reshape(STA1_Testing_X(1:52,:) + 1i*STA1_Testing_X(53:104,:), nUSC, nSym, N_Test_Frames);
     STA1_Testing_Y = eval(['STA_DNN_151015_test_y_',num2str(i)]);
     STA1_Testing_Y = reshape(STA1_Testing_Y(1:52,:) + 1i*STA1_Testing_Y(53:104,:), nUSC, nSym, N_Test_Frames);
     STA_DNN1_Y = eval(['STA_DNN_151015_corrected_y_',num2str(i)]);
     STA_DNN1_Y = reshape(STA_DNN1_Y(1:52,:) + 1i*STA_DNN1_Y(53:104,:), nUSC, nSym, N_Test_Frames);
     
     
     load(['D:\STA_DNN_151515_Results_' num2str(i),'.mat']);
     STA2_Testing_X = eval(['STA_DNN_151515_test_x_',num2str(i)]);
     STA2_Testing_X = reshape(STA2_Testing_X(1:52,:) + 1i*STA2_Testing_X(53:104,:), nUSC, nSym, N_Test_Frames);
     STA2_Testing_Y = eval(['STA_DNN_151515_test_y_',num2str(i)]);
     STA2_Testing_Y = reshape(STA2_Testing_Y(1:52,:) + 1i*STA2_Testing_Y(53:104,:), nUSC, nSym, N_Test_Frames);
     STA_DNN2_Y = eval(['STA_DNN_151515_corrected_y_',num2str(i)]);
     STA_DNN2_Y = reshape(STA_DNN2_Y(1:52,:) + 1i*STA_DNN2_Y(53:104,:), nUSC, nSym, N_Test_Frames);

     
     % Loading AE-DNN Results
     load(['D:\TRFI_DNN_151515_Results_' num2str(i),'.mat']);
     TRFI_Testing_X = eval(['TRFI_DNN_151515_test_x_',num2str(i)]);
     TRFI_Testing_X = reshape(TRFI_Testing_X(1:52,:) + 1i*TRFI_Testing_X(53:104,:), nUSC, nSym, N_Test_Frames);
     TRFI_Testing_Y = eval(['TRFI_DNN_151515_test_y_',num2str(i)]);
     TRFI_Testing_Y = reshape(TRFI_Testing_Y(1:52,:) + 1i*TRFI_Testing_Y(53:104,:), nUSC, nSym, N_Test_Frames);
     TRFI_DNN_Y = eval(['TRFI_DNN_151515_corrected_y_',num2str(i)]);
     TRFI_DNN_Y = reshape(TRFI_DNN_Y(1:52,:) + 1i*TRFI_DNN_Y(53:104,:), nUSC, nSym, N_Test_Frames);

 
    disp(['Running Simulation, SNR = ', num2str(EbN0dB(i))]);
    tic;
    for u = 1:N_Test_Frames
         c = Testing_Packets_Indices(u,1);        
         Phf(i)  = Phf(i)  + norm(True_Channels_Structure(:,:,c))^ 2;
        
        % AE-DNN 
        H_AE_DNN = DPA_DNN_Y(:,:,u);
        Err_AE_DNN (i) =  Err_AE_DNN (i) +  norm(H_AE_DNN - True_Channels_Structure(:,:,c)).^2;
        Equalized_OFDM_Symbols_AE_DNN =  Received_Symbols_FFT_Structure(dpositions ,:,c) ./ H_AE_DNN(dpositions,:);
        De_Mapped_AE_DNN = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_AE_DNN,M);
        Bits_AE_DNN      = zeros(nDSC,nSym,log2(M));
    
        % STA-DNN1
        H_STA_DNN1 = STA_DNN1_Y(:,:,u);
        Err_STA_DNN1 (i) =  Err_STA_DNN1 (i) +  norm(H_STA_DNN1 - True_Channels_Structure(:,:,c)).^2;
        Equalized_OFDM_Symbols_STA_DNN1 =  Received_Symbols_FFT_Structure(dpositions ,:,c) ./ H_STA_DNN1(dpositions,:);
        De_Mapped_STA_DNN1 = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_STA_DNN1,M);
        Bits_STA_DNN1      = zeros(nDSC,nSym,log2(M));
       
        
        % STA-DNN2
        H_STA_DNN2 = STA_DNN2_Y(:,:,u);
        Err_STA_DNN2 (i) =  Err_STA_DNN2 (i) +  norm(H_STA_DNN2 - True_Channels_Structure(:,:,c)).^2;
        Equalized_OFDM_Symbols_STA_DNN2 =  Received_Symbols_FFT_Structure(dpositions ,:,c) ./ H_STA_DNN2(dpositions,:);
        De_Mapped_STA_DNN2 = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_STA_DNN2,M);
        Bits_STA_DNN2      = zeros(nDSC,nSym,log2(M));
       
        % TRFI-DNN
        H_TRFI_DNN = TRFI_DNN_Y(:,:,u);
        Err_TRFI_DNN (i) =  Err_TRFI_DNN (i) +  norm(H_TRFI_DNN - True_Channels_Structure(:,:,c)).^2;
        Equalized_OFDM_Symbols_TRFI_DNN =  Received_Symbols_FFT_Structure(dpositions ,:,c) ./ H_TRFI_DNN(dpositions,:);
        De_Mapped_TRFI_DNN = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_TRFI_DNN,M);
        Bits_TRFI_DNN      = zeros(nDSC,nSym,log2(M));
       
        
        for b = 1 : nSym
           Bits_AE_DNN(:,b,:)    = de2bi(De_Mapped_AE_DNN(:,b));
           Bits_STA_DNN1(:,b,:)    = de2bi(De_Mapped_STA_DNN1(:,b));
           Bits_STA_DNN2(:,b,:)    = de2bi(De_Mapped_STA_DNN2(:,b));
           Bits_TRFI_DNN(:,b,:)    = de2bi(De_Mapped_TRFI_DNN(:,b));   
        end
  
        
         General_Block_De_Interleaved_Data_AE_DNN   = deintrlv(Bits_AE_DNN(:),Random_permutation_Vector);
         Matrix_De_Interleaved_Data_AE_DNN   = matintrlv(General_Block_De_Interleaved_Data_AE_DNN.',Interleaver_Columns,Interleaver_Rows).';
         Decoded_Bits_AE_DNN   = vitdec(Matrix_De_Interleaved_Data_AE_DNN,trellis,tbl,'trunc','hard');
         Bits_AE_DNN_Final = wlanScramble(Decoded_Bits_AE_DNN,scramInit);
         ber_AE_DNN = biterr(Bits_AE_DNN_Final,TX_Bits_Stream_Structure(:,c));
         Ber_AE_DNN(i)   = Ber_AE_DNN(i) + ber_AE_DNN;
  
        General_Block_De_Interleaved_Data_STA_DNN1   = deintrlv(Bits_STA_DNN1(:),Random_permutation_Vector);
        Matrix_De_Interleaved_Data_STA_DNN1   = matintrlv(General_Block_De_Interleaved_Data_STA_DNN1.',Interleaver_Columns,Interleaver_Rows).';
        Decoded_Bits_STA_DNN1   = vitdec(Matrix_De_Interleaved_Data_STA_DNN1,trellis,tbl,'trunc','hard');
        Bits_STA_DNN_Final1 = wlanScramble(Decoded_Bits_STA_DNN1,scramInit);
        ber_STA_DNN1 = biterr(Bits_STA_DNN_Final1,TX_Bits_Stream_Structure(:,c));
        Ber_STA_DNN1(i)   = Ber_STA_DNN1(i) + ber_STA_DNN1;
        
        General_Block_De_Interleaved_Data_STA_DNN2   = deintrlv(Bits_STA_DNN2(:),Random_permutation_Vector);
        Matrix_De_Interleaved_Data_STA_DNN2   = matintrlv(General_Block_De_Interleaved_Data_STA_DNN2.',Interleaver_Columns,Interleaver_Rows).';
        Decoded_Bits_STA_DNN2   = vitdec(Matrix_De_Interleaved_Data_STA_DNN2,trellis,tbl,'trunc','hard');
        Bits_STA_DNN_Final2 = wlanScramble(Decoded_Bits_STA_DNN2,scramInit);
        ber_STA_DNN2 = biterr(Bits_STA_DNN_Final2,TX_Bits_Stream_Structure(:,c));
        Ber_STA_DNN2(i)   = Ber_STA_DNN2(i) + ber_STA_DNN2;
        
        
         General_Block_De_Interleaved_Data_TRFI_DNN   = deintrlv(Bits_TRFI_DNN(:),Random_permutation_Vector);
         Matrix_De_Interleaved_Data_TRFI_DNN   = matintrlv(General_Block_De_Interleaved_Data_TRFI_DNN.',Interleaver_Columns,Interleaver_Rows).';
         Decoded_Bits_TRFI_DNN   = vitdec(Matrix_De_Interleaved_Data_TRFI_DNN,trellis,tbl,'trunc','hard');
         Bits_TRFI_DNN_Final = wlanScramble(Decoded_Bits_TRFI_DNN,scramInit);
         ber_TRFI_DNN = biterr(Bits_TRFI_DNN_Final,TX_Bits_Stream_Structure(:,c));
         Ber_TRFI_DNN(i)   = Ber_TRFI_DNN(i) + ber_TRFI_DNN;
        
    end
   toc;
end
Phf = Phf ./ N_Test_Frames;

ERR_AE_DNN = Err_AE_DNN ./ (N_Test_Frames * Phf); 
BER_AE_DNN = Ber_AE_DNN/ (N_Test_Frames * nSym * nDSC * nBitPerSym);


ERR_STA_DNN1 = Err_STA_DNN1 ./ (N_Test_Frames * Phf); 
BER_STA_DNN1 = Ber_STA_DNN1 ./ (N_Test_Frames * nSym * nDSC * nBitPerSym);


ERR_STA_DNN2 = Err_STA_DNN2 ./ (N_Test_Frames * Phf); 
BER_STA_DNN2 = Ber_STA_DNN2 ./ (N_Test_Frames * nSym * nDSC * nBitPerSym);

ERR_TRFI_DNN = Err_TRFI_DNN ./ (N_Test_Frames * Phf); 
BER_TRFI_DNN = Ber_TRFI_DNN/ (N_Test_Frames * nSym * nDSC * nBitPerSym);




