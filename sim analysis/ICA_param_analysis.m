clear all, close all, clc

%% Notes about the file
%{
-Data is from the sin0450 (exp. no 4) simulation.
-Simulation executed with 'v7' data on sim.ipynb (Jupyter Notebook).
-ICA is done within sim.ipynb.
-Both parts (vicinity & interaction) of original are in S($) domain.
-Wdist = Winte + Waero + Werrm.
%}

%% 1. Initializing data
fprintf("Loading data...\n");
data = csvread('ICA_params.csv',1,0); % dist data

%% Additional disturbance
Fs = length(data);                      % sampling frequency 
% Ts = 1/Fs;                              % sampling period at time step
% dt = 0 : Ts : 1-Ts;                     % signal duration
% 
% f_wind = 150;                           % frequency of the wind disturb.
% wind = 0.5*sin(2*pi*f_wind*dt);         % an arbitrary wind function
% data = data + transpose(wind);          % data with arbitrary wind

%% 2. Selecting necessary parts of data
S0_dist_vicinity = data(1:215,10);      % vicinity period
S1_dist_vicinity = data(1:215,11);
S2_dist_vicinity = data(1:215,12);
lv = length(S0_dist_vicinity);          % length of the signal

S0_dist_interact = data(220:906,10);    % interaction period
S1_dist_interact = data(220:906,11);
S2_dist_interact = data(220:906,12);
li = length(S0_dist_interact);          % length of the signal

%% 3. Saving data
% fprintf("\nWriting data to .csv files...\n");
% csvwrite('S0_dist_vicin.csv',S0_dist_vicinity);
% fprintf("S0 (dist) vicinity written to a .csv\n");
% csvwrite('S1_dist_vicin.csv',S1_dist_vicinity);
% fprintf("S1 (dist) vicinity written to a .csv\n");
% csvwrite('S2_dist_vicin.csv',S2_dist_vicinity);
% fprintf("S2 (dist) vicinity written to a .csv\n");
% 
% csvwrite('S0_dist_inter.csv',S0_dist_interact);
% fprintf("S0 (dist) interaction written to a .csv\n");
% csvwrite('S1_dist_inter.csv',S1_dist_interact);
% fprintf("S1 (dist) interaction written to a .csv\n");
% csvwrite('S2_dist_inter.csv',S2_dist_interact);
% fprintf("S2 (dist) interaction written to a .csv\n");

%% 4. Fourier transform
% vicinity data
fprintf("\nPerforming FFT of S($) components\n");
NFFTv = length(S0_dist_vicinity);           % length of the signal
NFFT_pow_v = 2^nextpow2(NFFTv);             % length of signal in power of 2
XFFTv = Fs*(0:NFFT_pow_v/2-1)/NFFT_pow_v;   % frequency vector (x-axis)

S0_vicin_fft = fft(S0_dist_vicinity,NFFT_pow_v);   % converting data to freq. domain
S0_vicin_sampled = S0_vicin_fft(1:NFFT_pow_v/2);   % sampling the data
S1_vicin_fft = fft(S1_dist_vicinity,NFFT_pow_v);
S1_vicin_sampled = S1_vicin_fft(1:NFFT_pow_v/2);
S2_vicin_fft = fft(S2_dist_vicinity,NFFT_pow_v);
S2_vicin_sampled = S2_vicin_fft(1:NFFT_pow_v/2);

% interaction data
NFFTi = length(S0_dist_interact);           % length of the signal
NFFT_pow_i = 2^nextpow2(NFFTi);             % length of signal in power of 2
XFFTi = Fs*(0:NFFT_pow_i/2-1)/NFFT_pow_i;   % frequency vector (x-axis)

S0_inte_fft = fft(S0_dist_interact,NFFT_pow_i);   % converting data to freq. domain
S0_inte_sampled = S0_inte_fft(1:NFFT_pow_i/2);    % sampling the data
S1_inte_fft = fft(S1_dist_interact,NFFT_pow_i);
S1_inte_sampled = S1_inte_fft(1:NFFT_pow_i/2);
S2_inte_fft = fft(S2_dist_interact,NFFT_pow_i);
S2_inte_sampled = S2_inte_fft(1:NFFT_pow_i/2);

%% 5. Plotting data
fprintf("\nPlotting vicinity period S($)");
figure(1), clf(1),
subplot(3,1,1),
plot(1:lv,S0_dist_vicinity), title('S0 vicinity');
subplot(3,1,2),
plot(1:lv,S1_dist_vicinity), title('S1 vicinity');
subplot(3,1,3),
plot(1:lv,S2_dist_vicinity), title('S2 vicinity');

fprintf("\nPlotting interaction period S($)");
figure(2), clf(2),
subplot(3,1,1),
plot(1:li,S0_dist_interact), title('S0 interaction');
subplot(3,1,2),
plot(1:li,S1_dist_interact), title('S1 interaction');
subplot(3,1,3),
plot(1:li,S2_dist_interact), title('S2 interaction');

fprintf("\nPlotting vicinity period S($) after FFT");
figure(3), clf(3),
subplot(3,1,1),
plot(XFFTv,abs(S0_vicin_sampled)), title('FFT of S0 vicinity'),
xlabel('Frequency [Hz]'),
ylabel('Amplitude [-]');
subplot(3,1,2),
plot(XFFTv,abs(S1_vicin_sampled)), title('FFT of S1 vicinity'),
xlabel('Frequency [Hz]'),
ylabel('Amplitude [-]');
subplot(3,1,3),
plot(XFFTv,abs(S2_vicin_sampled)), title('FFT of S2 vicinity'),
xlabel('Frequency [Hz]'),
ylabel('Amplitude [-]');

fprintf("\nPlotting interaction period S($) after FFT\n\n");
figure(4), clf(4),
subplot(3,1,1),
plot(XFFTi,abs(S0_inte_sampled)), title('FFT of S0 interaction'),
xlabel('Frequency [Hz]'),
ylabel('Amplitude [-]');
subplot(3,1,2),
plot(XFFTi,abs(S1_inte_sampled)), title('FFT of S1 interaction'),
xlabel('Frequency [Hz]'),
ylabel('Amplitude [-]');
subplot(3,1,3),
plot(XFFTi,abs(S2_inte_sampled)), title('FFT of S2 interaction'),
xlabel('Frequency [Hz]'),
ylabel('Amplitude [-]');