function [H_TRFI, Equalized_OFDM_Symbols] = TRFI(he_LS_Preamble ,y_r, Kset, rp2, pp, ppositions, mod, nUSC, nSym)
%,URS
%[subcarriers, OFDM_Symbols] = size(y_r(Kset,:)); 
%H_TRFI0 = he_LS_Preamble;
%URS = 0;
%Received_OFDM_Symbols = y_r(Kset,:);
H_TRFI = zeros(nUSC, nSym);
Equalized_OFDM_Symbols = zeros(nUSC, nSym);
for i = 1:nSym
    if(i == 1)
        % Step 1: Equalization.
        Equalized_OFDM_Symbol = y_r(Kset,i)./ he_LS_Preamble;
        Equalized_OFDM_Symbols(:,i) = Equalized_OFDM_Symbol;
       
        % Step 2: Constructing Data Pilot.
        De_Equalized_OFDM_Symbol = wlanClosestReferenceSymbol(Equalized_OFDM_Symbol,mod);
        De_Equalized_OFDM_Symbol(ppositions,:) = [1;1;1;-1]; 
        % Step 3: LS Estimation.
        Initial_Channel_Estimate = y_r(Kset,i)./ De_Equalized_OFDM_Symbol;
        
        % Step 4: Equalization and Demapping
        E1_0 = real(rp2 ./ Initial_Channel_Estimate);
        E2_0 = pp;
        X1_0 = wlanClosestReferenceSymbol(E1_0,'BPSK');
        X2_0 =  E2_0;
        
        % Add this step just to ensure that the pilots indices are included
        % in the trusted subcarriers set
        X1_0(ppositions,1) = 0;
        X2_0(ppositions,1) = 0;
        
        
        % Step 5: Comparison
        equal_indices = find(X1_0 == X2_0);
        unequal_indices = find(X1_0 ~= X2_0);
        H_TRFI(equal_indices,1) = Initial_Channel_Estimate(equal_indices,1);
     
        % Interpolation of unrealiable subcarriers 
         H_TRFI(unequal_indices,1) = interp1(equal_indices,H_TRFI(equal_indices,1),unequal_indices,'spline'); 
          
         %URS = URS + size(unequal_indices,1);
 
    elseif (i > 1)
        % Step 1: Equalization 
        Equalized_OFDM_Symbol = y_r(Kset,i)./ H_TRFI(:,i-1);
        Equalized_OFDM_Symbols(:,i) = Equalized_OFDM_Symbol;
        % Step 2: Constructing Data Pilot.
        De_Equalized_OFDM_Symbol = wlanClosestReferenceSymbol(Equalized_OFDM_Symbol,mod);
        De_Equalized_OFDM_Symbol(ppositions,:) = [1;1;1;-1]; 
        % Step 3: LS Estimation.
        Initial_Channel_Estimate = y_r(Kset,i)./ De_Equalized_OFDM_Symbol;
        % Step 4: Equalization and Demapping
        E1 =  y_r(Kset,i - 1)./ Initial_Channel_Estimate;
        E2 =  y_r(Kset,i - 1)./ H_TRFI(:,i-1);
        
        X1 =  wlanClosestReferenceSymbol(E1,mod);
        X2 =  wlanClosestReferenceSymbol(E2,mod);
        
        X1(ppositions,1) = 0;
        X2(ppositions,1) = 0;
        
        % Step 5: Comparison
        equal_indices = find(X1 == X2);
        unequal_indices = find(X1 ~= X2);
        
        H_TRFI(equal_indices,i) = Initial_Channel_Estimate(equal_indices,1);
        
        % Interpolation of unrealiable subcarriers 
        H_TRFI(unequal_indices,i) = interp1(equal_indices,H_TRFI(equal_indices,i),unequal_indices,'spline'); 
        %URS = URS + size(unequal_indices,1);
    end
end

end

