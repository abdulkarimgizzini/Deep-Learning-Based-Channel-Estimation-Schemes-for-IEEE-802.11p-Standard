function [H_CDP, Equalized_OFDM_Symbols] = CDP(he_LS_Preamble ,y_r, Kset, rp2, pp, mod, nUSC, nSym)
%[subcarriers, OFDM_Symbols] = size(y_r(Kset,:));
% H_CDP0 = he_LS_Preamble;
%Received_OFDM_Symbols = y_r(Kset,:);
H_CDP = zeros(nUSC, nSym);
Equalized_OFDM_Symbols = zeros(nUSC, nSym);
for i = 1:nSym
    if(i == 1)
        % Step 1: Equalization. First, the ith received symbol, Received_OFDM_Symbols(i,k) is equalized by the channel estimate, 
        % of the previous symbol. For i = 1, CDP(0) is the LS estimate at
        % the preamble.
        Equalized_OFDM_Symbol = y_r(Kset,i)./ he_LS_Preamble;
        Equalized_OFDM_Symbols(:,i) = Equalized_OFDM_Symbol;
       
        % Step 2: Constructing Data Pilot. Equalized_OFDM_Symbol is demodulated to obtain De_Equalized_OFDM_Symbol.
        De_Equalized_OFDM_Symbol = wlanClosestReferenceSymbol(Equalized_OFDM_Symbol,mod);
      
        % Step 3: LS Estimation. the ith received symbol is divided by  De_Equalized_OFDM_Symbol to obtain the initial channel estimate
        Initial_Channel_Estimate = y_r(Kset,i)./ De_Equalized_OFDM_Symbol;
        
        
        % Step 4: Equalization and Demapping
        E1_0 = real(rp2 ./ Initial_Channel_Estimate);
        E2_0 = pp;
        X1_0 = wlanClosestReferenceSymbol(E1_0,'BPSK');
        X2_0 =  E2_0;
        
        % Step 5: Comparison
        equal_indices = find(X1_0 == X2_0);
        unequal_indices = find(X1_0 ~= X2_0);
        
        H_CDP(equal_indices,1) = Initial_Channel_Estimate(equal_indices,1);
        H_CDP(unequal_indices,1) = he_LS_Preamble(unequal_indices,1); 
        
 
    elseif (i > 1)
        % Step 1: Equalization 
        Equalized_OFDM_Symbol = y_r(Kset,i)./ H_CDP(:,i-1);
        Equalized_OFDM_Symbols(:,i) = Equalized_OFDM_Symbol;
        % Step 2: Constructing Data Pilot.
        De_Equalized_OFDM_Symbol = wlanClosestReferenceSymbol(Equalized_OFDM_Symbol,mod);
        % Step 3: LS Estimation.
        Initial_Channel_Estimate = y_r(Kset,i)./ De_Equalized_OFDM_Symbol;
        % Step 4: Equalization and Demapping
        E1 =  y_r(Kset,i - 1)./ Initial_Channel_Estimate;
        E2 =  y_r(Kset,i - 1)./ H_CDP(:,i-1);
        
        X1 =  wlanClosestReferenceSymbol(E1,mod);
        X2 =  wlanClosestReferenceSymbol(E2,mod);
        
        % Step 5: Comparison
        equal_indices = find(X1 == X2);
        unequal_indices = find(X1 ~= X2);
        
        H_CDP(equal_indices,i) = Initial_Channel_Estimate(equal_indices,1);
        H_CDP(unequal_indices,i) = H_CDP(unequal_indices,i - 1);
    end
end

end

