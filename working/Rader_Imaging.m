
% タ(

clear all
close all

%% --------row--------------------------------------

load('comon_parameter.mat')
load('rowsignal.mat')

%% -----------------------------------------------

% w = hamming(length(S(1,:))).';   %  
% W = ones(length(S(:,1)),1)*w;    % 
% S = S.*W;   % 

%% -----鰚-----------------------------------------

fs = Tinterval^-1;   % □ [Hz]
N_fft = 2^13;        % FFT
r_max = 100;         % ヲ [m]

Image = fft(S, N_fft, 2);     % (

f = 0:fs/(N_fft-1):fs;        % □イ
r = c*dT*f/(2*bandwide);      % □イイ

x = u*v;  %イ

cut = find(r<= r_max+0.1,1, 'last');
r = r(1:cut);
Image  = Image (:,1:cut);

% ヲ
figure
clims = [-60 0];
imagesc(r, x, 20*log10(abs(Image )/max(max(abs(Image )))), clims)
xlabel('Range [m]')
ylabel('Runing Distance [m]')
xlim([0 r_max])
colorbar

