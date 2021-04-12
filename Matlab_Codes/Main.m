clc;clearvars;close all; warning('off','all');
ch_func = Channel_functions();
%% --------OFDM Parameters - Given in IEEE 802.11p Spec--
ofdmBW                 = 10 * 10^6 ;     % OFDM bandwidth (Hz)
nFFT                   = 64;             % FFT size 
nDSC                   = 48;             % Number of data subcarriers
nPSC                   = 4;              % Number of pilot subcarriers
nZSC                   = 12;             % Number of zeros subcarriers
nUSC                   = nDSC + nPSC;    % Number of total used subcarriers
K                      = nUSC + nZSC;    % Number of total subcarriers
nSym                   = 100;            % Number of OFDM symbols within one frame
deltaF                 = ofdmBW/nFFT;    % Bandwidth for each subcarrier - include all used and unused subcarriers 
Tfft                   = 1/deltaF;       % IFFT or FFT period = 6.4us
Tgi                    = Tfft/4;         % Guard interval duration - duration of cyclic prefix - 1/4th portion of OFDM symbols = 1.6us
Tsignal                = Tgi+Tfft;       % Total duration of BPSK-OFDM symbol = Guard time + FFT period = 8us
K_cp                   = nFFT*Tgi/Tfft;  % Number of symbols allocated to cyclic prefix 
pilots_locations       = [8,22,44,58].'; % Pilot subcarriers positions
pilots                 = [1 1 1 -1].';
data_locations         = [2:7, 9:21, 23:27, 39:43, 45:57, 59:64].'; % Data subcarriers positions
ppositions             = [7,21, 32,46].';                           % Pilots positions in Kset
dpositions             = [1:6, 8:20, 22:31, 33:45, 47:52].';        % Data positions in Kset
% Pre-defined preamble in frequency domain
dp = [ 0  0 0 0 0 0 +1 +1 -1 -1 +1  +1 -1  +1 -1 +1 +1 +1 +1 +1 +1 -1 -1 +1 +1 -1 +1 -1 +1 +1 +1 +1 0 +1 -1 -1 +1 +1 -1 +1 -1 +1 -1 -1 -1 -1 -1 +1 +1 -1 -1 +1 -1 +1 -1 +1 +1 +1 +1 0 0 0 0 0];
Ep                     = 1;              % pramble power per sample
dp                     = fftshift(dp);   % Shift zero-frequency component to center of spectrum    
predefined_preamble    = dp;
Kset                   = find(dp~=0);               % set of allocated subcarriers                  
Kon                    = length(Kset);              % Number of active subcarriers
dp                     = sqrt(Ep)*dp.';
xp                     = ifft(dp);
xp_cp                  = [xp(end-K_cp+1:end); xp];  % Adding CP to the time domain preamble
preamble_80211p        = repmat(xp_cp,1,2);         % IEEE 802.11p preamble symbols (tow symbols)
% to be used in the preamble based MMSE Channel Estimation
preamble_diag          = diag(predefined_preamble(Kset)) * diag(predefined_preamble(Kset).');
% STA Channel Estimation Scheme Parameters
alpha = 2;
Beta = 2;
w = 1 / (2*Beta +1);
lambda = -Beta:Beta;
%% ------ Bits Modulation Technique------------------------------------------
mod                       = 'QPSK';
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
%% ---------Scrambler Parameters---------------------------------------------
scramInit                 = 93; % As specidied in IEEE 802.11p Standard [1011101] in binary representation
%% ---------Convolutional Coder Parameters-----------------------------------
constlen                  = 7;
trellis                   = poly2trellis(constlen,[171 133]);
tbl                       = 34;
rate                      = 1/2;
%% -------Interleaver Parameters---------------------------------------------
% Matrix Interleaver
Interleaver_Rows          = 16;
Interleaver_Columns       = (nBitPerSym * nDSC * nSym) / Interleaver_Rows;
% General Block Interleaver
Random_permutation_Vector = randperm(nBitPerSym*nDSC*nSym); % Permutation vector
%% -----------------Vehicular Channel Model Parameters--------------------------
ChType                    = 'VTV_UC';               % Channel model
L_Tabs                    = 12;
fs                        = K*deltaF;               % Sampling frequency in Hz, here case of 802.11p with 64 subcarriers and 156250 Hz subcarrier spacing
fc                        = 5.9e9;                  % Carrier Frequecy in Hz.
v                         = 48;                     % Moving speed of user in km/h
c                         = 3e8;                    % Speed of Light in m/s
fD                        = (v/3.6)/c*fc;           % Doppler freq in Hz
plotFlag                  = 0;                      % 1 to display the channel frequency response
rchan                     = ch_func.GenFadingChannel(ChType, fD, fs);
release(rchan);
init_seed = 22;
%% ---------Bit to Noise Ratio------------------%
EbN0dB                    = 0:5:30;         % bit to noise ratio
SNR_p                     = EbN0dB + 10*log10(K/nDSC) + 10*log10(K/(K + K_cp)) + 10*log10(nBitPerSym) + 10*log10(rate); % converting to symbol to noise ratio
SNR_p                     = SNR_p.';
%snr_p = Ep/KN0 => N0 = Ep/(K*snr_p)
N0 = Ep/K*10.^(-SNR_p/10);
%% Simulation Parameters 
N_CH                    = 1000; % number of channel realizations
N_SNR                   = length(SNR_p); % SNR length

% Normalized mean square error (NMSE) vectors
Err_LS_Preamble         = zeros(N_SNR,1);
Err_Perfect_Preamble    = zeros(N_SNR,1);
Err_STA                 = zeros(N_SNR,1);
Err_Initial             = zeros(N_SNR,1);
Err_CDP                 = zeros(N_SNR,1);
Err_TRFI                = zeros(N_SNR,1);
Err_MMSE_VP             = zeros(N_SNR,1);

% Bit error rate (BER) vectors
Ber_Ideal               = zeros(N_SNR,1);
Ber_LS                  = zeros(N_SNR,1);
Ber_Perfect_Preamble    = zeros(N_SNR,1);
Ber_STA                 = zeros(N_SNR,1);
Ber_Initial             = zeros(N_SNR,1);
Ber_CDP                 = zeros(N_SNR,1);
Ber_TRFI                = zeros(N_SNR,1);
Ber_MMSE_VP             = zeros(N_SNR,1);
Ber_CPI                 = zeros(N_SNR,1);
% average channel power E(|hf|^2)
Phf_H_Total             = zeros(N_SNR,1);
%% Simulation Loop
for n_snr = 1:N_SNR
    disp(['Running Simulation, SNR = ', num2str(EbN0dB(n_snr))]);
    tic;    
    TX_Bits_Stream_Structure                = zeros(nDSC * nSym  * nBitPerSym *rate, N_CH);
    Received_Symbols_FFT_Structure          = zeros(Kon,nSym, N_CH);
    True_Channels_Structure                 = zeros(Kon, nSym, N_CH);
    DPA_Structure                          = zeros(Kon, nSym, N_CH);
    STA_Structure                          = zeros(Kon, nSym, N_CH);
    TRFI_Structure                          = zeros(Kon, nSym, N_CH);
    
    for n_ch = 1:N_CH % loop over channel realizations
        
        % Bits Stream Generation 
        Bits_Stream_Coded = randi(2, nDSC * nSym  * nBitPerSym * rate,1)-1;
        % Data Scrambler 
        scrambledData = wlanScramble(Bits_Stream_Coded,scramInit);
        % Convolutional Encoder
        dataEnc = convenc(scrambledData,trellis);
        % Interleaving
        % Matrix Interleaving
        codedata = dataEnc.';
        Matrix_Interleaved_Data = matintrlv(codedata,Interleaver_Rows,Interleaver_Columns).';
        % General Block Interleaving
        General_Block_Interleaved_Data = intrlv(Matrix_Interleaved_Data,Random_permutation_Vector);
        % Bits Mapping: M-QAM Modulation
        TxBits_Coded = reshape(General_Block_Interleaved_Data,nDSC , nSym  , nBitPerSym);
        % Gray coding goes here
        TxData_Coded = zeros(nDSC ,nSym);
        for m = 1 : nBitPerSym
           TxData_Coded = TxData_Coded + TxBits_Coded(:,:,m)*2^(m-1);
        end
        % M-QAM Modulation
         Modulated_Bits_Coded  =1/sqrt(Pow) * qammod(TxData_Coded,M);
         % Check the power of Modulated bits. it must be equal to 1
         avgPower = mean (abs (Modulated_Bits_Coded).^ 2);
         % OFDM Frame Generation
         OFDM_Frame_Coded = zeros(K,nSym);
         OFDM_Frame_Coded(data_locations,:) = Modulated_Bits_Coded;
         OFDM_Frame_Coded(pilots_locations,:) = repmat(pilots,1,nSym);
         % Taking FFT, the term (nFFT/sqrt(nDSC)) is for normalizing the power of transmit symbol to 1 
         IFFT_Data_Coded = (K/sqrt(nUSC))*ifft(OFDM_Frame_Coded);
         % checking the power of the transmit signal (it has to be 1 after normalization)
         power_Coded = var(IFFT_Data_Coded(:)) + abs(mean(IFFT_Data_Coded(:)))^2; 
         % Appending cylic prefix
         CP_Coded = IFFT_Data_Coded((K - K_cp +1):K,:);
         IFFT_Data_CP_Coded = [CP_Coded; IFFT_Data_Coded];
         % Appending preamble symbol 
         %IFFT_Data_CP_Preamble_Coded = [xp_cp IFFT_Data_CP_Coded];
         IFFT_Data_CP_Preamble_Coded = [ preamble_80211p IFFT_Data_CP_Coded];
         power_transmitted = sqrt((K+K_cp)/K) * var(IFFT_Data_CP_Preamble_Coded(:)) + abs(mean(IFFT_Data_CP_Preamble_Coded(:)))^2; 

        % ideal estimation
        [ h, y ] = ch_func.ApplyChannel( rchan, IFFT_Data_CP_Preamble_Coded, K_cp);
        release(rchan);
        rchan.Seed = rchan.Seed+1;
        
        yp = y((K_cp+1):end,1:2);
        y  = y((K_cp+1):end,3:end);
        
        yFD = (sqrt(nUSC)/K)*fft(y);
        yfp = fft(yp); % FD preamble
        
        
        h = h((K_cp+1):end,:);
        hf = fft(h); % Fd channel
        hf  = hf(:,3:end);

      
        
        Phf_H_Total(n_snr) = Phf_H_Total(n_snr) + norm(hf(Kset))^2;
        %add noise
        noise_preamble = sqrt(K*N0(n_snr))*ch_func.GenRandomNoise([K,2], 1);
        yfp_r = yfp +  noise_preamble;
        noise_OFDM_Symbols = sqrt(K*N0(n_snr))*ch_func.GenRandomNoise([K,size(yFD,2)], 1);
        y_r   = yFD + noise_OFDM_Symbols;
       
      
       %% Channel Estimation
       % IEEE 802.11p LS Estimate at Preambles
       he_LS_Preamble = ((yfp_r(Kset,1) + yfp_r(Kset,2))./(2.*predefined_preamble(Kset).'));
       H_LS = repmat(he_LS_Preamble,1,nSym);
       err_LS_Preamble = norm(H_LS - hf(Kset,:))^2;
       Err_LS_Preamble (n_snr) = Err_LS_Preamble (n_snr) + err_LS_Preamble;
       
       %STA Channel Estimation
       [H_STA, Equalized_OFDM_Symbols_STA] = STA(he_LS_Preamble ,y_r, Kset,mod, nUSC, nSym, ppositions, alpha, w, lambda);
       err_H_STA = norm(H_STA - hf(Kset,:))^2;
       Err_STA(n_snr) = Err_STA(n_snr) + err_H_STA;
       
       
       %CDP Channel Estimation
        [H_CDP, Equalized_OFDM_Symbols_CDP] = CDP(he_LS_Preamble([1:6, 8:20, 22:31, 33:45, 47:52].',1) ,y_r, data_locations, yfp_r(data_locations,2), predefined_preamble(1,data_locations).', mod, nDSC, nSym);
        err_H_CDP = norm(H_CDP - hf(data_locations,:))^2;
        Err_CDP(n_snr) = Err_CDP(n_snr) + err_H_CDP;
      
       %TRFI Channel Estimation ,URS
       [H_TRFI, Equalized_OFDM_Symbols_TRFI] = TRFI(he_LS_Preamble ,y_r, Kset, yfp_r(Kset,2), predefined_preamble(1,Kset).', ppositions, mod, nUSC, nSym);
       err_H_TRFI = norm(H_TRFI - hf(Kset,:))^2;
       Err_TRFI(n_snr) = Err_TRFI(n_snr) + err_H_TRFI;
    

      
       %MMSE + Virtual Pilots
       noise_power_OFDM_Symbols = var(noise_OFDM_Symbols);
       [H_MMSE_VP, Equalized_OFDM_Symbols_MMSE_VP] = MMSE_Vitual_Pilots(he_LS_Preamble ,y_r, Kset,mod, ppositions, dpositions, noise_power_OFDM_Symbols);
       err_H_MMSE_VP = norm(H_MMSE_VP - hf(Kset,:))^2;
       Err_MMSE_VP(n_snr) = Err_MMSE_VP(n_snr) + err_H_MMSE_VP;

       % Initial Channel Estimation
       [H_Initial, Equalized_OFDM_Symbols_Initial] = Initial_Channel_Estimation(he_LS_Preamble ,y_r, Kset, ppositions, mod, nUSC, nSym);
       err_H_Initial = norm(H_Initial - hf(Kset,:))^2;
       Err_Initial(n_snr) = Err_Initial(n_snr) + err_H_Initial;

  
       
       % Equalization
     y_Ideal = y_r(data_locations ,:) ./ hf(data_locations,:); %Ideal
     y_LS = y_r(data_locations ,:)./ H_LS(dpositions,:); % LS


     % QAM - DeMapping
       De_Mapped_Ideal = qamdemod(sqrt(Pow) * y_Ideal,M);
       De_Mapped_LS = qamdemod(sqrt(Pow) * y_LS,M);
       De_Mapped_STA = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_STA(dpositions,:),M);
       De_Mapped_Initial = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_Initial(dpositions,:),M); 
       De_Mapped_CDP = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_CDP ,M);
       De_Mapped_TRFI = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_TRFI(dpositions,:),M);
       De_Mapped_MMSE_VP = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_MMSE_VP(dpositions,:),M);
      




          % Bits Extraction
           Bits_Ideal = zeros(nDSC,nSym,log2(M));
           Bits_LS     = zeros(nDSC,nSym,log2(M));
           Bits_STA         = zeros(nDSC,nSym,log2(M));
           Bits_Initial      = zeros(nDSC,nSym,log2(M));
           Bits_CDP         = zeros(nDSC,nSym,log2(M));
           Bits_TRFI         = zeros(nDSC,nSym,log2(M));
           Bits_MMSE_VP         = zeros(nDSC,nSym,log2(M));
       
        
          for b = 1 : nSym
                   Bits_Ideal(:,b,:) = de2bi(De_Mapped_Ideal(:,b));
                   Bits_LS(:,b,:) = de2bi(De_Mapped_LS(:,b)); 
                   Bits_STA(:,b,:) = de2bi(De_Mapped_STA(:,b)); 
                   Bits_Initial(:,b,:) = de2bi(De_Mapped_Initial(:,b)); 
                   Bits_CDP(:,b,:) = de2bi(De_Mapped_CDP(:,b));
                   Bits_TRFI(:,b,:) = de2bi(De_Mapped_TRFI(:,b));
                   Bits_MMSE_VP(:,b,:) = de2bi(De_Mapped_MMSE_VP(:,b));                  
          end           
          
          
 
          % De-Interleaving
          % General Block De-Interleaving
          General_Block_De_Interleaved_Data_Ideal = deintrlv(Bits_Ideal(:),Random_permutation_Vector);
          General_Block_De_Interleaved_Data_LS    = deintrlv(Bits_LS(:),Random_permutation_Vector);
          General_Block_De_Interleaved_Data_STA   = deintrlv(Bits_STA(:),Random_permutation_Vector);
          General_Block_De_Interleaved_Data_Initial   = deintrlv(Bits_Initial(:),Random_permutation_Vector);
          General_Block_De_Interleaved_Data_CDP   = deintrlv(Bits_CDP(:),Random_permutation_Vector);
          General_Block_De_Interleaved_Data_TRFI   = deintrlv(Bits_TRFI(:),Random_permutation_Vector);
          General_Block_De_Interleaved_Data_MMSE_VP   = deintrlv(Bits_MMSE_VP(:),Random_permutation_Vector);
          

           
           % Matrix De-Interleaving
          Matrix_De_Interleaved_Data_Ideal = matintrlv(General_Block_De_Interleaved_Data_Ideal.',Interleaver_Columns,Interleaver_Rows).';
          Matrix_De_Interleaved_Data_LS    = matintrlv(General_Block_De_Interleaved_Data_LS.',Interleaver_Columns,Interleaver_Rows).';
           Matrix_De_Interleaved_Data_STA   = matintrlv(General_Block_De_Interleaved_Data_STA.',Interleaver_Columns,Interleaver_Rows).';
           Matrix_De_Interleaved_Data_Initial   = matintrlv(General_Block_De_Interleaved_Data_Initial.',Interleaver_Columns,Interleaver_Rows).';
          Matrix_De_Interleaved_Data_CDP   = matintrlv(General_Block_De_Interleaved_Data_CDP.',Interleaver_Columns,Interleaver_Rows).';
           Matrix_De_Interleaved_Data_TRFI   = matintrlv(General_Block_De_Interleaved_Data_TRFI.',Interleaver_Columns,Interleaver_Rows).';
           Matrix_De_Interleaved_Data_MMSE_VP   = matintrlv(General_Block_De_Interleaved_Data_MMSE_VP.',Interleaver_Columns,Interleaver_Rows).';
           


           % Viterbi decoder
           Decoded_Bits_Ideal = vitdec(Matrix_De_Interleaved_Data_Ideal,trellis,tbl,'trunc','hard');
           Decoded_Bits_LS    = vitdec(Matrix_De_Interleaved_Data_LS,trellis,tbl,'trunc','hard');
            Decoded_Bits_STA   = vitdec(Matrix_De_Interleaved_Data_STA,trellis,tbl,'trunc','hard');
            Decoded_Bits_Initial   = vitdec(Matrix_De_Interleaved_Data_Initial,trellis,tbl,'trunc','hard');
            Decoded_Bits_CDP   = vitdec(Matrix_De_Interleaved_Data_CDP,trellis,tbl,'trunc','hard');
            Decoded_Bits_TRFI   = vitdec(Matrix_De_Interleaved_Data_TRFI,trellis,tbl,'trunc','hard');
            Decoded_Bits_MMSE_VP   = vitdec(Matrix_De_Interleaved_Data_MMSE_VP,trellis,tbl,'trunc','hard');
           

           
           
           % De-scrambler Data
             Bits_Ideal_Final = wlanScramble(Decoded_Bits_Ideal,scramInit);
             Bits_LS_Final = wlanScramble(Decoded_Bits_LS,scramInit);
            Bits_STA_Final = wlanScramble(Decoded_Bits_STA,scramInit);
            Bits_Initial_Final = wlanScramble(Decoded_Bits_Initial,scramInit);
            Bits_CDP_Final = wlanScramble(Decoded_Bits_CDP,scramInit);
            Bits_TRFI_Final = wlanScramble(Decoded_Bits_TRFI,scramInit);
            Bits_MMSE_VP_Final = wlanScramble(Decoded_Bits_MMSE_VP,scramInit);
           

            
           
           % BER and FER Calculation
           ber_Ideal  = biterr(Bits_Ideal_Final,Bits_Stream_Coded);
           
           
           ber_LS     = biterr(Bits_LS_Final,Bits_Stream_Coded);
           
           
            ber_STA = biterr(Bits_STA_Final,Bits_Stream_Coded);
            ber_Initial = biterr(Bits_Initial_Final,Bits_Stream_Coded);    
            ber_CDP = biterr(Bits_CDP_Final,Bits_Stream_Coded); 
            ber_TRFI = biterr(Bits_TRFI_Final,Bits_Stream_Coded);
            ber_MMSE_VP = biterr(Bits_MMSE_VP_Final,Bits_Stream_Coded);
             

            
            
            Ber_Ideal (n_snr) = Ber_Ideal (n_snr) + ber_Ideal;
            Ber_LS (n_snr)    = Ber_LS (n_snr) + ber_LS;
            Ber_STA(n_snr)    = Ber_STA(n_snr) + ber_STA;
            Ber_Initial(n_snr)    = Ber_Initial(n_snr) + ber_Initial;
            Ber_TRFI(n_snr)   = Ber_TRFI(n_snr) + ber_TRFI;
            Ber_CDP(n_snr)    = Ber_CDP(n_snr) + ber_CDP;
            Ber_MMSE_VP(n_snr)   = Ber_MMSE_VP(n_snr) + ber_MMSE_VP;
           
            
          TX_Bits_Stream_Structure(:, n_ch) = Bits_Stream_Coded;
          Received_Symbols_FFT_Structure(:,:,n_ch) = y_r(Kset,:);
          True_Channels_Structure(:,:,n_ch) = hf(Kset,:);
          DPA_Structure(:,:,n_ch)  = H_Initial;
          STA_Structure(:,:,n_ch)  = H_STA;
          TRFI_Structure(:,:,n_ch)  = H_TRFI;          
    end 
    
    save(['D:\Simulation_' num2str(n_snr)],...
           'TX_Bits_Stream_Structure',...
            'Received_Symbols_FFT_Structure',...
            'True_Channels_Structure',...
            'DPA_Structure','STA_Structure','TRFI_Structure');
    toc;
end 
%% Bit Error Rate (BER)
BER_Ideal             = Ber_Ideal /(N_CH * nSym * nDSC * nBitPerSym);
BER_LS                = Ber_LS / (N_CH * nSym * nDSC * nBitPerSym);
BER_STA               = Ber_STA / (N_CH * nSym * nDSC * nBitPerSym);
BER_Initial           = Ber_Initial / (N_CH * nSym * nDSC * nBitPerSym);
BER_CDP               = Ber_CDP / (N_CH * nSym * nDSC * nBitPerSym);
BER_TRFI              = Ber_TRFI / (N_CH * nSym * nDSC * nBitPerSym);
BER_MMSE_VP           = Ber_MMSE_VP / (N_CH * nSym * nDSC* nBitPerSym);
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
grid on;
legend([p1(1),p2(1),p3(1),p4(1),p5(1),p6(1),p7(1)],{'Perfect Channel','LS','STA','Initial Channel','CDP','TRFI','MMSE-VP'});
xlabel('SNR(dB)');
ylabel('Bit Error Rate (BER)');

%% Normalized Mean Square Error
Phf_H       = Phf_H_Total/(N_CH);
ERR_LS      = Err_LS_Preamble / (Phf_H * N_CH * nSym);  
ERR_STA     = Err_STA / (Phf_H * N_CH * nSym);
ERR_Initial     = Err_Initial / (Phf_H * N_CH * nSym);
ERR_CDP     = Err_CDP / (Phf_H * N_CH * nSym);
ERR_TRFI    = Err_TRFI / (Phf_H * N_CH * nSym);
ERR_MMSE_VP = Err_MMSE_VP / (Phf_H * N_CH * nSym);

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
grid on;
legend([p1(1),p2(1),p3(1),p4(1),p5(1),p6(1)],{'LS','STA','Initial Channel','CDP','TRFI','MMSE-VP'});
xlabel('SNR(dB)');
ylabel('Normalized Mean Sqaure Error (NMSE)');

save(['D:\Simulation_variables'],'mod','Kset','Random_permutation_Vector','fD','ChType');