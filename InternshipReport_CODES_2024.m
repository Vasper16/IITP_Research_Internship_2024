% 1.Unit Step Signal
t = -5:0.01:5;
u = double(t >= 0);
plot(t, u);
title('Unit Step Signal');
xlabel('Time');
ylabel('Amplitude');
grid on;

% 2.Ramp Signal
t = -5:0.01:5;
ramp = t .* (t >= 0);
plot(t, ramp);
title('Ramp Signal');
xlabel('Time');
ylabel('Amplitude');
grid on;

% 3.Exponential Signal
t = -5:0.01:5;
A = 1;
alpha = -1;
exp_signal = A * exp(alpha * t);
plot(t, exp_signal);
title('Exponential Signal');
xlabel('Time');
ylabel('Amplitude');
grid on;

% 4.Sinusoidal Signal
t = -5:0.01:5;
f = 1;  % Frequency in Hz
sin_wave = sin(2*pi*f*t);
plot(t, sin_wave);
title('Sinusoidal Signal');
xlabel('Time');
ylabel('Amplitude');
grid on;

% 5.Impulse Signal
t = -5:0.01:5;
impulse = zeros(size(t));
impulse(abs(t) < 0.01) = 1;
plot(t, impulse);
title('Impulse Signal (Approximated)');
xlabel('Time');
ylabel('Amplitude');
grid on;

% 6.BPSK Constellation
N = 1000; % Number of bits
data = randi([0 1], 1, N); % Random binary data

bpskMod = 2*data - 1; % BPSK Mapping: 0→-1, 1→1
scatterplot(bpskMod);
title('BPSK Constellation Diagram');
xlabel('In-phase'); ylabel('Quadrature');
grid on;

% 7.QPSK Constellation
N = 1000;
data = randi([0 3], 1, N); % 2 bits per symbol

qpskMod = pskmod(data, 4, pi/4); % QPSK with phase offset pi/4
scatterplot(qpskMod);
title('QPSK Constellation Diagram');
xlabel('In-phase'); ylabel('Quadrature');
grid on;

% 8. 8-PSK Constellation
N = 1000;
data = randi([0 7], 1, N); % 3 bits per symbol

psk8Mod = pskmod(data, 8, pi/8); % 8-PSK with offset
scatterplot(psk8Mod);
title('8-PSK Constellation Diagram');
xlabel('In-phase'); ylabel('Quadrature');
grid on;

% 9. 4-QAM 
N = 1000;
data = randi([0 3], 1, N); 

qam4Mod = qammod(data, 4);
scatterplot(qam4Mod);
title('4-QAM Constellation Diagram');
xlabel('In-phase'); ylabel('Quadrature');
grid on;

% 10. 16-QAM Constellation
N = 1000;
data = randi([0 15], 1, N); % 4 bits per symbol

qam16Mod = qammod(data, 16);
scatterplot(qam16Mod);
title('16-QAM Constellation Diagram');
xlabel('In-phase'); ylabel('Quadrature');
grid on;

% 11. 64-QAM Constellation
N = 1000;
data = randi([0 63], 1, N); % 6 bits per symbol

qam64Mod = qammod(data, 64);
scatterplot(qam64Mod);
title('64-QAM Constellation Diagram');
xlabel('In-phase'); ylabel('Quadrature');
grid on;

% 12. Noise Modeling Example using AWGN
fs = 1000;               % Sampling frequency
t = 0:1/fs:1-1/fs;       % Time vector
signal = sin(2*pi*50*t); % Example sinusoidal signal

snr = 10;                % Signal-to-noise ratio in dB
noisy_signal = awgn(signal, snr, 'measured');

figure;
plot(t, signal, 'b', t, noisy_signal, 'r--');
legend('Original Signal', 'Noisy Signal');
xlabel('Time (s)');
ylabel('Amplitude');
title('Noise Modeling using AWGN');


% 13. CHANNEL EFFECTS: AWGN vs RAYLEIGH for BPSK
clc; clear; close all;

% Simulation Parameters
N = 1e5;                      % Number of bits
snr_dB = 0:2:20;              % SNR range in dB
ber_awgn = zeros(size(snr_dB));
ber_rayleigh = zeros(size(snr_dB));

% Generate random binary data
data = randi([0 1], 1, N);

% BPSK Modulation: 0 -> -1, 1 -> +1
tx = 2*data - 1;

for i = 1:length(snr_dB)
    %% --- AWGN Channel ---
    rx_awgn = awgn(tx, snr_dB(i), 'measured');  % Add white Gaussian noise
    rx_awgn_bits = rx_awgn > 0;                 % Hard decision
    ber_awgn(i) = sum(rx_awgn_bits ~= data)/N;  % BER calculation

    %% --- Rayleigh Fading Channel ---
    h = (randn(1, N) + 1j*randn(1, N))/sqrt(2);  % Rayleigh fading (complex)
    faded_signal = h .* tx;                     % Apply channel
    rx_rayleigh = awgn(faded_signal, snr_dB(i), 'measured'); % Add noise
    rx_eq = real(rx_rayleigh ./ h);             % Equalize (channel compensation)
    rx_rayleigh_bits = rx_eq > 0;               % Hard decision
    ber_rayleigh(i) = sum(rx_rayleigh_bits ~= data)/N;  % BER calculation
end

%% --- PLOTTING THE RESULTS ---
figure;
semilogy(snr_dB, ber_awgn, 'bo-', 'LineWidth', 2);
hold on;
semilogy(snr_dB, ber_rayleigh, 'r*-', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('BER vs SNR for BPSK under AWGN and Rayleigh Fading Channels');
legend('AWGN Channel', 'Rayleigh Channel');
axis([0 20 1e-5 1]);


% 14. BER vs SNR for OFDM using PSK over AWGN
% Parameters
N = 64;                 % Number of subcarriers
cp_len = 16;            % Cyclic prefix length
num_sym = 1000;         % Number of OFDM symbols
SNR_dB = 0:2:20;        % SNR range
ber = zeros(size(SNR_dB));

% Total number of bits
total_bits = N * num_sym;

% Generate random bits
bits = randi([0 1], total_bits, 1);

% BPSK modulation: 0 -> -1, 1 -> +1
bpsk = 2*bits - 1;

% Reshape into [N x num_sym] for OFDM
tx_data = reshape(bpsk, N, num_sym);

% IFFT (time domain signal)
ifft_data = ifft(tx_data);

% Add cyclic prefix
cp_data = [ifft_data(end - cp_len + 1:end, :); ifft_data];

% Serialize signal
tx_signal = cp_data(:);

% Loop over SNR values
for i = 1:length(SNR_dB)
    % Transmit over AWGN
    rx_signal = awgn(tx_signal, SNR_dB(i), 'measured');

    % Reshape back to OFDM frames
    rx_matrix = reshape(rx_signal, N + cp_len, num_sym);

    % Remove cyclic prefix
    rx_no_cp = rx_matrix(cp_len + 1:end, :);

    % FFT (frequency domain)
    rx_freq = fft(rx_no_cp);

    % BPSK demodulation
    rx_bits = real(rx_freq(:)) > 0;

    % BER calculation
    ber(i) = sum(rx_bits ~= bits) / total_bits;
end

% Plot BER vs SNR
figure;
semilogy(SNR_dB, ber, 'r-o', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('BER vs SNR for OFDM using BPSK over AWGN');
legend('OFDM-BPSK');

% OFDM Simulation for 5G-like scenario
clc; clear; close all;

N = 64;                 % Number of subcarriers
CP_len = 16;            % Length of cyclic prefix
M = 4;                  % QPSK Modulation
num_symbols = 100;      % Number of OFDM symbols

% Transmitter
data = randi([0 M-1], N, num_symbols);     % Random data
mod_data = pskmod(data, M, pi/M);          % QPSK modulation
ifft_data = ifft(mod_data, N);             % IFFT for OFDM

% Add cyclic prefix
ofdm_signal = [ifft_data(end-CP_len+1:end, :); ifft_data];

% Channel: AWGN + Rayleigh
h = (1/sqrt(2))*(randn(N+CP_len, num_symbols) + 1i*randn(N+CP_len, num_symbols));
rx_signal = ofdm_signal .* h;
rx_signal = awgn(rx_signal, 20, 'measured');

% Receiver
rx_signal_no_cp = rx_signal(CP_len+1:end, :);
rx_fft = fft(rx_signal_no_cp, N);

% Equalization
rx_eq = rx_fft ./ h(CP_len+1:end, :);
rx_demod = pskdemod(rx_eq, M, pi/M);

% BER
[numErr, ber] = biterr(data, rx_demod);
fprintf('OFDM BER: %f\n', ber);

% Simple MIMO Simulation with 2x2 system
clc; clear; close all;

N = 1000;
M = 4;  % QPSK

data_tx1 = randi([0 M-1], 1, N);
data_tx2 = randi([0 M-1], 1, N);

x1 = pskmod(data_tx1, M, pi/M);
x2 = pskmod(data_tx2, M, pi/M);

H = (1/sqrt(2))*(randn(2,N) + 1i*randn(2,N)); % 2x2 Rayleigh channel
X = [x1; x2];
N0 = 0.1*(randn(2,N) + 1i*randn(2,N));

Y = sum(H .* X, 1) + N0; % Received signal

% Simplified ZF detection
H_eff = sum(H, 2);
X_hat = Y ./ sum(H, 1);
x1_hat = pskdemod(X_hat, M, pi/M);

ber = sum(data_tx1 ~= x1_hat)/N;
fprintf('MIMO BER (simplified): %f\n', ber);


% Channel Models Comparison: AWGN, Rayleigh, Rician
clc; clear; close all;

N = 1e5;
data = randi([0 1], 1, N);
bpsk = 2*data - 1;

EbN0 = 10;
snr = EbN0;

% AWGN
awgn_sig = awgn(bpsk, snr, 'measured');
awgn_detected = awgn_sig > 0;
ber_awgn = sum(data ~= awgn_detected)/N;

% Rayleigh
ray = (1/sqrt(2))*(randn(1,N) + 1i*randn(1,N));
ray_sig = ray .* bpsk;
ray_rx = awgn(ray_sig, snr, 'measured') ./ ray;
ray_detected = real(ray_rx) > 0;
ber_rayleigh = sum(data ~= ray_detected)/N;

% Rician
K = 5;
LOS = ones(1,N);
NLOS = (1/sqrt(2))*(randn(1,N) + 1i*randn(1,N));
h_rician = sqrt(K/(K+1))*LOS + sqrt(1/(K+1))*NLOS;
rician_sig = h_rician .* bpsk;
rician_rx = awgn(rician_sig, snr, 'measured') ./ h_rician;
rician_detected = real(rician_rx) > 0;
ber_rician = sum(data ~= rician_detected)/N;

fprintf('BER AWGN: %e\n', ber_awgn);
fprintf('BER Rayleigh: %e\n', ber_rayleigh);
fprintf('BER Rician: %e\n', ber_rician);



