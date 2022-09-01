function [H_MMSE_VP, Equalized_OFDM_Symbols] = MMSE_Vitual_Pilots(he_LS_Preamble ,y_r, Kset,mod, ppositions,dpositions, noise_power_OFDM_Symbols)
[subcarriers, OFDM_Symbols] = size(y_r(Kset,:)); 
H_MMSE_VP0 = he_LS_Preamble;
Received_OFDM_Symbols = y_r(Kset,:);
H_MMSE_VP = zeros(subcarriers, OFDM_Symbols);
Equalized_OFDM_Symbols = zeros(subcarriers, OFDM_Symbols);
P = zeros(subcarriers, OFDM_Symbols);
H = zeros(subcarriers, OFDM_Symbols);
for i = 1:OFDM_Symbols
    if(i == 1)
        
		% Step 1: Equalization
        Equalized_OFDM_Symbol = Received_OFDM_Symbols(:,i)./ H_MMSE_VP0;
        Equalized_OFDM_Symbols(:,i) = Equalized_OFDM_Symbol;
        
		% Step 2: Demodulation (Constructing Data Pilots)
        De_Equalized_OFDM_Symbol = wlanClosestReferenceSymbol(Equalized_OFDM_Symbol,mod);
        De_Equalized_OFDM_Symbol(ppositions,:) = [1;1;1;-1]; 
        
		% Step 3: LS Channel Estimation
        Initial_Channel_Estimate = Received_OFDM_Symbols(:,i)./ De_Equalized_OFDM_Symbol;
        H(:,i) = Initial_Channel_Estimate; 
        
		% Step 4: Constructing the reference signal
        P(:,i) = [Initial_Channel_Estimate(7,1); Initial_Channel_Estimate(21,1); Initial_Channel_Estimate(dpositions,1); Initial_Channel_Estimate(32,1); Initial_Channel_Estimate(46,1)];
        
		% Step 5: Calculate the cross-correlation matrices between the initial channel estimate vector, 
        % and the reference signal’s channel P1. the auto-correlation
        % matrix of reference signal P1.
        R_HP = zeros(subcarriers,subcarriers);
        R_PP = zeros(subcarriers,subcarriers);
        for m = 1:i
            R_HP = R_HP + H(:,m) * P(:,m)';
            R_PP = R_PP + P(:,m) * P(:,m)';
        end
        R_HP = (1/i) .* R_HP;
        R_PP = (1/i) .* R_PP;
        
		% Step 6: Final MMSE_VP Channel Estimate
        H_MMSE_VP(:,1) = (R_HP / (R_PP + eye(subcarriers) * noise_power_OFDM_Symbols(1,1))) * P(:,i);
   

    elseif (i > 1)
        
		% Step 1: Equalization
        Equalized_OFDM_Symbol = Received_OFDM_Symbols(:,i)./ H_MMSE_VP(:,i-1);
        Equalized_OFDM_Symbols(:,i) = Equalized_OFDM_Symbol;
        
		% Step 2: Demodulation (Constructing Data Pilots)
        De_Equalized_OFDM_Symbol = wlanClosestReferenceSymbol(Equalized_OFDM_Symbol,mod);
        De_Equalized_OFDM_Symbol(ppositions,:) = [1;1;1;-1];         
        
		%  Step 3: LS Channel Estimation
        Initial_Channel_Estimate = Received_OFDM_Symbols(:,i)./ De_Equalized_OFDM_Symbol;
         H(:,i) = Initial_Channel_Estimate; 
        
		% Step 4: Constructing the reference signal
        P(:,i) = [Initial_Channel_Estimate(7,1); Initial_Channel_Estimate(21,1); Initial_Channel_Estimate(dpositions,1); Initial_Channel_Estimate(32,1); Initial_Channel_Estimate(46,1)];
        
		% Step 5: Calculate the cross-correlation matrices between the initial channel estimate vector, 
        % and the reference signal’s channel P1. the auto-correlation
        % matrix of reference signal P1.
        R_HP = zeros(subcarriers,subcarriers);
        R_PP = zeros(subcarriers,subcarriers);
        for m = 1:i
            R_HP = R_HP + H(:,m) * P(:,m)';
            R_PP = R_PP + P(:,m) * P(:,m)';
        end
        R_HP = (1/i) .* R_HP;
        R_PP = (1/i) .* R_PP;
        
		% Step 6: Final MMSE_VP Channel Estimate
        H_MMSE_VP(:,i) = (R_HP / (R_PP + eye(subcarriers) * noise_power_OFDM_Symbols(1,i))) * P(:,i);
    end
end

end

