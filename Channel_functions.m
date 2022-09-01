function func = Channel_functions()
%% --------------- Memeber Variables -------------------------------------- 
% It can be used to store variable, for example constant.
%% --------------- Memeber functions Declaration --------------------------
% reference to functions
%gain power profile, L number of taps, NR number of realizations
%H = DummyChannel(gain, L, NR)
func. DummyChannel = @DummyChannel;
%v = GenRandomNoise(siz, 0)
func.GenRandomNoise = @GenRandomNoise;
% fs sampling frequency 
% fD Doppler frequency
%rchan = GenFadingChannel( ChType, fD, fs)
func.GenFadingChannel = @GenFadingChannel;
func.GenFadingChannel2 = @GenFadingChannel2;
% He estimatated channel after removing CP.
%[ He, Y ] = ApplyChannel( rchan, X, Ncp)
func.ApplyChannel = @ApplyChannel;
func.ApplyChannel2 = @ApplyChannel2;
func.UpdateSeed   = @UpdateSeed;
func.GetSeed      = @GetSeed;
func.SetSeed      = @SetSeed;
func.PreambleAutoCorrection     = @PreambleAutoCorrection;
%% --------------- Including of library function --------------------------
% call of other structures 
%% --------------- Implementation -----------------------------------------
% function implementation
function H = DummyChannel(gain, NR)
	 L = numel(gain);
        H = repmat(gain, 1, NR).*GenRandomNoise([L, NR], 1);
    end
function v = GenRandomNoise(siz, N0)
v = sqrt(N0/2) * (randn(siz)+1j*randn(siz));
end
function s = UpdateSeed(rchan)
release(rchan);
rchan.Seed = rchan.Seed + 1;
s = rchan.Seed;
end

function ChannelSeed = GetSeed(rchan)

    ChannelSeed = rchan.Seed;
end

function SetSeed(rchan,seed)
 release(rchan);
    rchan.Seed = seed;
end
function [ He, Y ] = ApplyChannel( rchan, X, Ncp)
%release(rchan);
%rchan.Seed = rchan.Seed+1; % change realization
[Ns, NB] = size(X);
D = zeros(Ns,NB);
% Estimate the channel appling a pulse
D(Ncp+1,:) = 1;
He = zeros(size(D));
for nb=1:NB
    He(:,nb) = step(rchan, D(:,nb));
end
% reset the channel to first state which correspond to the estimation
reset(rchan);
y = step(rchan, X(:));
Y = reshape(y,Ns, NB);
end
function [He, Hf_Preamble] = PreambleAutoCorrection(rchan,Ncp,Nsc,Ns)
release(rchan);
D = zeros(Nsc,Ns);
% Estimate the channel appling a pulse
D(Ncp+1,:) = 1;
He = zeros(size(D));
for nb=1:Ns
    He(:,nb) = step(rchan, D(:,nb));
end
H = fft(He);
Hf_Preamble = H(:,1);
end

function [ He, Y,Y2 ] = ApplyChannel2( rchan, X,X2, Ncp)
release(rchan);
%rchan.Seed = rchan.Seed+1; % change realization
[Ns, NB] = size(X);
D = zeros(Ns,NB);
% Estimate the channel appling a pulse
D(Ncp+1,:) = 1;
He = zeros(size(D));
for nb=1:NB
    He(:,nb) = step(rchan, D(:,nb));
end
% reset the channel to first state which correspond to the estimation
reset(rchan);
y = step(rchan, X(:));
Y = reshape(y,Ns, NB);

reset(rchan);
y2 = step(rchan, X2(:));
Y2 = reshape(y2,Ns, NB);
end

% ChType = EPA, EVA, TGn, FLAT
function rchan = GenFadingChannel( ChType, fD, fs)
%GENFADINGCHANNEL Summary of this function goes here
%   Detailed explanation goes here
switch ChType
    case 'PedA'
        PathDelays = 1e-9.*[0 110 190 410];
        avgPathGains = [0 -9.7 -19.2 -22.8];
    case 'PedB'
        PathDelays = 1e-9.*[0 200 800 1200 2300 3700];
        avgPathGains = [0 -0.9 -4.9 -8.0 -7.8 -23.9];  
    case 'VehA'
        PathDelays = 1e-9.*[0 310 710 1090 1730 2510];
        avgPathGains = [0 -1 -9 -10 -15 -20];
    case 'EVA'
        PathDelays = 1e-9.*[0 30 150 310 370 710 1090 1730 2510];
        avgPathGains = [0 -1.5 -1.4 -3.6 -0.6 -9.1 -7.0 -12.0 -16.9];
    case 'EPA'
        PathDelays = 1e-9.*[0 30 70 ];
        avgPathGains = [0 -1 -2  ];
    case 'TU'
        PathDelays = 1e-9.*[0 50 120 200 230 500 1600 2300 5000];
        avgPathGains = [-1 -1 -1 0 0 0 -3 -5 -7];
    case 'VehB'
        PathDelays = 1e-9.*[0 300 8900 12900 17100 20000];
        avgPathGains = [-2.5 0 -12.8 -10.0 -25.2 -16];
    case 'HT'
        PathDelays = 1e-9.*[0 356 441 528 546 609 625 842 916 941 15000 16172 16492 16876 16882 16978 17615 17827 17849 18016];
        avgPathGains = [-3.6 -8.9 -10.2 -11.5 -11.8 -12.7 -13.0 -16.2 -17.3 -17.7 -17.6 -22.7 -24.1 -25.8 -25.8 -26.2 -29.0 -29.9 -30.0 -30.7];
    case 'ETU'
        PathDelays = 1e-9.*[0 217 512 514 517 674 882 1230 1287 1311 1349 1533 1535 1622 1818 1836 1884 1943 2048 2140];
        avgPathGains = [-5.7 -7.6 -10.1 -10.2 -10.2 -11.5 -13.4 -16.3 -16.9 -17.1 -17.4 -19.0 -19.0 -19.8 -21.5 -21.6 -22.1 -22.6 -23.5 -24.30];
    case 'RA'
        PathDelays = 1e-9.*[0 42 101 129 149 245 312 410 469 528];
        avgPathGains = [-5.2 -6.4 -8.4 -9.3 -10 -13.1 -15.3 -18.5 -20.4 -22.4];
    case 'IOA'
        PathDelays = 1e-9.*[0 50 110 170 290 310];
        avgPathGains = [0 -3 -10 -18 -26 -32];
    case 'IOB'
        PathDelays = 1e-9.*[0 100 200 300 500 700];
        avgPathGains = [0 -3.6 -7.2 -10.8 -18.0 -25.2];
        
    case 'Flat'
      PathDelays = 1e-9.*[0];
       avgPathGains = [0];% -3.6 -7.2 -10.8 -18.0 -25.2];
    %-------Channel Models Used in Vehicular communications----------------
    case 'RTV_UC_TRFI_Paper'
        PathDelays = 1e-9.*[0, 100, 200, 300, 400, 500, 600];
        avgPathGains = [0, -3.5, -5.1, -8, -10.9, -14, -21.5];
    case 'VTV_EX'
        PathDelays = 1e-9.*[0, 1, 2, 100, 101, 200, 201,202,300,301,302];
        avgPathGains = [0,0,0,-6.3,-6.3,-25.1,-25.1, -25.1,-22.7,-22.7,-22.7];
    case 'RTV_EX'
        PathDelays = 1e-9.*[0, 1, 2, 100, 101, 102, 200, 201, 300, 301, 400, 401];
        avgPathGains = [0, 0, 0, -9.3, -9.3, -9.3, -20.3, -20.3, -21.3, -21.3, -28.8,-28.8];    
    case 'VTV_SDWW'
        PathDelays = 1e-9.*[0, 1, 100, 101, 200, 300, 400, 401, 500, 600, 700, 701];
        avgPathGains = [0, 0, -11.2,-11.2,-19,-21.9, -25.3, -25.3, -24.4, -28.0, -26.1,-26.1];   
    case 'RTV_UC'
        PathDelays = 1e-9.*[0, 1, 2, 100, 101, 102, 200, 201, 300, 301, 500, 501];
        avgPathGains = [0, 0, 0, -11.5, -11.5, -11.5, -19.0, -19.0, -25.6, -25.6, -28.1,-28.1];
    case 'RTV_SUS'
        PathDelays = 1e-9.*[0, 1, 100, 101, 200, 201, 300, 301, 400, 500, 600, 700];
        avgPathGains = [0, 0, -9.3, -9.3, -14, -14, -18, -18, -19.4, -24.9, -27.5,-29.8];  
    case 'VTV_UC'
        PathDelays = 1e-9.*[0, 1, 100, 101, 102, 200, 201, 202, 300, 301, 400, 401];
        avgPathGains = [0, 0, -10, -10, -10, -17.8, -17.8, -17.8, -21.1, -21.1, -26.3,-26.3];
    case 'HIPERLAN_E'
        PathDelays = 1e-9.*[0, 10, 20, 40, 70, 100, 140, 190, 240, 320, 430, 560, 710, 880, 1070, 1280, 1510, 1760];
        avgPathGains = [-4.9, -5.1, -5.2, -0.8, -1.3, -1.9, -0.3, -1.2, -2.1, 0.0, -1.9, -2.8, -5.4, -7.3, -10.6, -13.4,-17.4,-20.9];   
    otherwise
        error('Channel model unknown');
end
rchan = comm.RayleighChannel('SampleRate',fs, ...
    'PathDelays',PathDelays, ...
    'AveragePathGains',avgPathGains, ...
    'MaximumDopplerShift',fD,...
    'RandomStream','mt19937ar with seed', ...
    'Seed',22);
end

function rchan2 = GenFadingChannel2(ChType, fs)
    switch ChType
     case 'R-LOS'
        p_dB = [0 -14 -17];
        tau = [0 83 183]*1e-9;
        dpl_shft = [0 492 -295];
    case 'UA-LOS'
        p_dB = [0 -8 -10 -15];
        tau = [0 117 183 333]*1e-9;
        dpl_shft = [0 236 -157 492];
    case 'C-NLOS'
        p_dB = [0 -3 -5 -10];
        tau = [0 267 400 533]*1e-9;
        dpl_shft = [0 295 -98 591];
    case 'H-LOS'
        p_dB = [0 -10 -15 -20];
        tau = [0 100 167 500]*1e-9;
        dpl_shft = [0 689 -492 886];
    case 'H-NLOS'
        p_dB = [0 -2 -5 -7];
        tau = [0 200 433 700]*1e-9;
        dpl_shft = [0 689 -492 886];
    case 'R-LOS-ENH'
        p_dB = [0 -12 -15];
        tau = [0 84 183]*1e-9;
        dpl_shft = [0 94 -1176];
    case 'UA-LOS-ENH'
        p_dB = [0 -11 -13 -15];
        tau = [0 222 334 533]*1e-9;
        dpl_shft = [0 224 1173 588];
    case 'C-NLOS-ENH'
        p_dB = [0 -3 -4 -7 -15];
        tau = [0 220 266 475 630]*1e-9;
        dpl_shft = [0 -142 -542 -155 320];
    case 'H-LOS-ENH'
        p_dB = [0 -11 -13 -17];
        tau = [0 167 433 600]*1e-9;
        dpl_shft = [0 1941 -1176 -391];
    case 'H-NLOS-ENH'
        p_dB = [0 -2 -5 -7 -15];
        tau = [0 100 500 867 1152]*1e-9;
        dpl_shft = [0 50 1157 -2352 1573];
    end
    
    % Use a very large K-factor for 1st tap (Static - pure Rician), zero for remaining taps (pure Rayleigh)
RicianFactor = [1e12 zeros(1, length(tau) - 1)];

% Maximum Doppler shift for all paths (identical)
fd = max(abs(dpl_shft));

% Initialize Doppler spectrum cell
doppler_spec = cell(size(dpl_shft));

% First tap is static (zero offset)
doppler_spec{1} = doppler('Asymmetric Jakes', [-.02 .02]);

% Remaining taps follow an Asymmetiric Jakes distribution (or "Half-Bathtub")
for ii = 2:length(dpl_shft)
    doppler_spec{ii} = doppler('Asymmetric Jakes', sort([0 dpl_shft(ii)/fd]));
end

% Create channel object
rchan2 = comm.MIMOChannel (...
    'KFactor',                  RicianFactor, ...
    'DirectPathDopplerShift', 	zeros(size(RicianFactor)), ...
    'DirectPathInitialPhase', 	zeros(size(RicianFactor)), ...
    'DopplerSpectrum',          doppler_spec, ...
    'SampleRate',               fs, ...
    'PathDelays',               tau, ...
    'AveragePathGains',         p_dB, ...
    'MaximumDopplerShift',      fd, ...
    'FadingDistribution',         'Rician', ...
    'NormalizePathGains',         true, ...
    'NormalizeChannelOutputs',    false, ...
    'TransmitCorrelationMatrix',  eye(1), ...
    'ReceiveCorrelationMatrix',   eye(1), ...
    'RandomStream','mt19937ar with seed', 'Seed',22);     
end
%% --------------- END of Implementation ----------------------------------
end
