%% 功能：单人呼吸心跳原始数据采集与MATLAB处理
%% 基于平台 IWR1642EVM+DCA1000
%% 思考：如何将IWR1843EVM+DCA1000采集得到的数据移植到这份代码上
%% ========================================================================
clc;clear;close all;
%% =========================================================================
%% 读取数据部分
%numADCSamples = 200; % number of ADC samples per chirp (原始IWR1642)
numADCSamples = 256; % number of ADC samples per chirp (AWR1843)
numADCBits = 16;     % number of ADC bits per sample
numRX = 4;           % number of receivers
numLanes = 2;        % do not change. number of lanes is always 2
isReal = 0;          % set to 1 if real only data, 0 if complex data0
chirpLoop = 128;      % numLoops=64, 2TX TDM -> 128 chirps/frame

%% 雷达参数设置
%Fs=4e6;             %ADC采样率 (原始IWR1642)
Fs=10e6;             %ADC采样率 (AWR1843)
c=3*1e8;             %光速
ts=numADCSamples/Fs; %ADC采样时间
slope=70.006e12;        %调频斜率 (原始IWR1642)
%slope=29.982e12;     %调频斜率 (AWR1843, 20260317数据配置)

B_valid =ts*slope;  %有效带宽
detaR=c/(2*B_valid);%距离分辨率

%% 读取Bin文件
%Filename = 'xxx.bin';  %文件名 (原始)
%Filename = 'E:\NJU\Beamforming_for_Sensing\data\mmwavestudio\Front_0.58_static_70.006_pause\adc_data.bin';  %AWR1843数据
Filename = 'E:\NJU\Beamforming_for_Sensing\data\mmwavestudio\20260330_1Person_0.6m_0deg_Standing_Low\adc_data.bin';  %AWR1843数据
fid = fopen(Filename,'r');
adcDataRow = fread(fid, 'int16');
if numADCBits ~= 16
    l_max = 2^(numADCBits-1)-1;
    adcDataRow(adcDataRow > l_max) = adcDataRow(adcDataRow > l_max) - 2^numADCBits;
end
fclose(fid);

fileSize = size(adcDataRow, 1);%获取文件大小，单位为采样点数
PRTnum = fix(fileSize/(numADCSamples*numRX));%计算chirp数，文件大小除以每个chirp的采样点数（numADCSamples*numRX）
fileSize = PRTnum * numADCSamples*numRX;
adcData = adcDataRow(1:fileSize);

% real data reshape, filesize = numADCSamples*numChirps
if isReal
    numChirps = fileSize/numADCSamples/numRX;
    LVDS = zeros(1, fileSize);
    %create column for each chirp
    LVDS = reshape(adcData, numADCSamples*numRX, numChirps);
    %each row is data from one chirp
    LVDS = LVDS.';
else
    numChirps = fileSize/2/numADCSamples/numRX;     %含有实部虚部除以2=frame*loops*txnums=1200*64*2
    LVDS = zeros(1, fileSize/2);
    %combine real and imaginary part into complex data
    %read in file: 2I is followed by 2Q
    counter = 1;
    for i=1:4:fileSize-1
        LVDS(1,counter) = adcData(i) + sqrt(-1)*adcData(i+2);
        LVDS(1,counter+1) = adcData(i+1)+sqrt(-1)*adcData(i+3); counter = counter + 2;
    end

    % create column for each chirp
    LVDS = reshape(LVDS, numADCSamples*numRX, numChirps);%[256*4,chrips]
    %each row is data from one chirp
    LVDS = LVDS.';%转置[chrips,256*4]
end

%% 重组数据
adcData = zeros(numRX,numChirps*numADCSamples);%[4,153600*256]
for row = 1:numRX
    for i = 1:numChirps
        adcData(row, (i-1)*numADCSamples+1:i*numADCSamples) = LVDS(i, (row-1)*numADCSamples+1:row*numADCSamples);
    end
end

%% =========================================================================
%% 数字波束形成（DBF）部分 - 只接收 17 度方向的信号
%% =========================================================================
% 步骤1：提取全部 4 路 RX 数据，整理为 [numRX, numADCSamples, numChirps]
rx_all = zeros(numRX, numADCSamples, numChirps);
for rx = 1:numRX
    rx_all(rx, :, :) = reshape(adcData(rx, :), numADCSamples, numChirps);
end

% 步骤2：分离 TX0 和 TX2 的 chirp（TDM-MIMO，奇数chirp=TX0，偶数chirp=TX2）
% 每帧128个chirp：1,3,5,...=TX0；2,4,6,...=TX2
tx0_data = rx_all(:, :, 1:2:end);  % [4, numADCSamples, numChirps/2]
tx2_data = rx_all(:, :, 2:2:end);  % [4, numADCSamples, numChirps/2]
numLoops = size(tx0_data, 3);      % = 1200*64 = 76800

% 步骤3：构建 8 元虚拟阵列（翻转RX顺序以匹配导向矢量定义方向）
% 参见 beamforming/build_virtual_array.m
% TX0+RX[3210] → 虚拟位置 [0,1,2,3]*d
% TX2+RX[3210] → 虚拟位置 [4,5,6,7]*d
va_data = zeros(8, numADCSamples, numLoops);
for loop = 1:numLoops
    tx0_rev = tx0_data(end:-1:1, :, loop);  % [4, numADCSamples]
    tx2_rev = tx2_data(end:-1:1, :, loop);  % [4, numADCSamples]
    va_data(:, :, loop) = [tx0_rev; tx2_rev];
end


%% 第一帧MVDR Range-Angle图（仅显示，不改变后续处理链路）
ra_ang_scan = -60:0.5:60;
RangFFT_ra = 512;
diag_load_alpha_ra = 0.1;

% 第一帧对应的loop索引（每帧64个loop）
loops_per_frame = 128;
frame1_idx = 1:loops_per_frame;

% 距离轴（米）
detaR_fft_ra = c / (2 * slope * ts) / RangFFT_ra * numADCSamples;
range_axis_ra = (0:RangFFT_ra-1) * detaR_fft_ra;

% 第一帧距离FFT数据：Xr_all [RangFFT_ra, 8, 64]
hann_win_ra = hann(numADCSamples);
Xr_all = zeros(RangFFT_ra, 8, loops_per_frame);
for ii = 1:loops_per_frame
    loop_id = frame1_idx(ii);
    x_in = va_data(:, :, loop_id).';                         % [numADCSamples, 8]
    x_win = x_in .* repmat(hann_win_ra, 1, 8);
    Xr_all(:, :, ii) = fft(x_win, RangFFT_ra, 1);           % [RangFFT_ra, 8]
end

% 第一帧慢时间去静态：每个[距离bin, 天线]减去64个loop均值
Xr_mean = mean(Xr_all, 3);
Xr_all = Xr_all - repmat(Xr_mean, 1, 1, loops_per_frame);

% 逐距离bin计算MVDR/Capon角谱
n_ra = (0:7).';
RA_mvdr = zeros(RangFFT_ra, numel(ra_ang_scan));
for rb = 2:RangFFT_ra
    Rxx_sum_ra = zeros(8, 8);
    for ii = 1:loops_per_frame
        x_bin = squeeze(Xr_all(rb, :, ii)).';               % [8, 1]
        Rxx_sum_ra = Rxx_sum_ra + (x_bin * x_bin');
    end
    Rxx_ra = Rxx_sum_ra / loops_per_frame;

    diag_load_ra = diag_load_alpha_ra * real(trace(Rxx_ra)) / 8;
    Rxx_loaded_ra = Rxx_ra + diag_load_ra * eye(8);

    for k = 1:numel(ra_ang_scan)
        a_ra = exp(-1j * pi * n_ra * sin(ra_ang_scan(k) * pi / 180));
        RA_mvdr(rb, k) = 1 / real(a_ra' * (Rxx_loaded_ra \ a_ra) + eps);
    end
end

% 转dB并显示
RA_mvdr_db = 10 * log10(RA_mvdr / max(RA_mvdr(:)) + eps);
figure('Color', 'w');
imagesc(ra_ang_scan, range_axis_ra, RA_mvdr_db);
axis xy;
colormap(jet);
colorbar;
caxis([-35 0]);
xlabel('角度（°）');
ylabel('距离（m）');
title('第一帧 MVDR Range-Angle 图');
ylim([0 5]);

% 8元ULA，天线间距 d=λ/2：a(θ)=exp(-j*π*n*sin(θ))
fc = 77e9;
lambda = c / fc;
d = lambda / 2;
n = (0:7)';

% ===== MVDR估角参数（可调） =====
target_range_m_mvdr = 0.5;    % 固定距离门（米）
RangFFT_bf = 512;              % 距离FFT点数（用于估角）
ang_scan = -60:0.2:60;         % 扫角范围
diag_load_alpha = 0.01;         % 对角加载系数

% 距离bin映射
detaR_fft_bf = c / (2 * slope * ts) / RangFFT_bf * numADCSamples;
target_bin_mvdr = round(target_range_m_mvdr / detaR_fft_bf) + 1;
target_bin_mvdr = max(2, min(target_bin_mvdr, RangFFT_bf));

% 在固定距离bin上累计协方差矩阵
hann_win_bf = hann(numADCSamples);
Rxx_sum = zeros(8, 8);
for loop = 1:numLoops
    x_in = va_data(:, :, loop).';                           % [numADCSamples, 8]
    x_win = x_in .* repmat(hann_win_bf, 1, 8);
    Xr = fft(x_win, RangFFT_bf, 1);                         % [RangFFT_bf, 8]
    x_bin = Xr(target_bin_mvdr, :).';                       % [8, 1]
    Rxx_sum = Rxx_sum + (x_bin * x_bin');
end
Rxx = Rxx_sum / max(numLoops, 1);

diag_load = diag_load_alpha * real(trace(Rxx)) / 8;
Rxx_loaded = Rxx + diag_load * eye(8);

% MVDR/Capon空间谱并估角
P_mvdr = zeros(size(ang_scan));
for k = 1:length(ang_scan)
    a = exp(-1j * pi * n * sin(ang_scan(k) * pi / 180));
    P_mvdr(k) = 1 / real(a' * (Rxx_loaded \ a) + eps);
end
[~, idx_pk] = max(P_mvdr);
theta_mvdr_deg = ang_scan(idx_pk);

fprintf('[MVDR-DOA] 目标距离: %.2f m -> 距离bin: %d（分辨率: %.4f m/bin）\n', ...
    target_range_m_mvdr, target_bin_mvdr, detaR_fft_bf);
fprintf('[MVDR-DOA] snapshots: %d, diag_load: %.3e\n', numLoops, diag_load);
fprintf('[MVDR-DOA] 估计角度 theta_mvdr: %.2f°\n', theta_mvdr_deg);

% MVDR角谱图（诊断）
P_mvdr_db = 10 * log10(P_mvdr / max(P_mvdr) + eps);
figure('Color','w');
plot(ang_scan, P_mvdr_db, 'm', 'LineWidth', 1.8); hold on;
xline(theta_mvdr_deg, '--k', sprintf('\\theta_{MVDR}=%.1f°', theta_mvdr_deg), 'LineWidth', 1.2);
grid on;
xlabel('角度（°）');
ylabel('归一化谱功率（dB）');
title('固定距离bin的MVDR角谱 — 左键点击选角度，右键结束');
xlim([ang_scan(1) ang_scan(end)]); ylim([-40 5]);

% ===== 交互式多角度选择 =====
fprintf('请在MVDR角谱图上左键点击选择目标角度，右键结束...\n');
selected_angles = [];
while true
    [x_click, ~, button] = ginput(1);
    if button == 3  % 右键结束
        break;
    end
    [~, idx_near] = min(abs(ang_scan - x_click));
    sel_angle = ang_scan(idx_near);
    if ~ismember(sel_angle, selected_angles)
        selected_angles(end+1) = sel_angle;
        xline(sel_angle, '--r', sprintf('%.1f°', sel_angle), 'LineWidth', 1.5);
        drawnow;
        fprintf('  已选角度: %.1f°\n', sel_angle);
    end
end
if isempty(selected_angles)
    fprintf('未选择任何角度，退出。\n');
    return;
end
fprintf('共选 %d 个角度: %s\n', numel(selected_angles), num2str(selected_angles, '%.1f  '));
%% ===== 多角度 CBF + 生命体征提取 =====
numAngles = numel(selected_angles);
RangFFT = 512;
hann_win = hann(numADCSamples);
fs = 20;        % 帧率（Hz），帧周期 50ms
T_frame = 0.05;
target_range_m = 0.5;  % 目标距离（米），可修改
detaR_fft = c / (2 * slope * ts) / RangFFT * numADCSamples;
max_num = round(target_range_m / detaR_fft) + 1;
max_num = max(2, min(max_num, RangFFT));
fprintf('目标距离: %.2f m → 距离bin: %d（分辨率: %.4f m/bin）\n', ...
    target_range_m, max_num, detaR_fft);

breath_counts  = zeros(1, numAngles);
heart_counts   = zeros(1, numAngles);
breath_signals = cell(1, numAngles);
heart_signals  = cell(1, numAngles);

for ai = 1:numAngles
    theta_sel = selected_angles(ai);
    theta_rad = theta_sel * pi / 180;
    steering_vec = exp(-1j * pi * n * sin(theta_rad));
    w = steering_vec / norm(steering_vec);

    % CBF波束形成
    bf_output = zeros(numADCSamples, numLoops);
    for loop = 1:numLoops
        bf_output(:, loop) = w' * va_data(:, :, loop);
    end

    % 每帧取第1个loop作为该帧代表
    process_adc = bf_output(:, 1:chirpLoop:end);
    numFrames = size(process_adc, 2);

    % 距离FFT（加Hann窗）
    adcdata_win = process_adc .* repmat(hann_win, 1, numFrames);
    fft_data = fft(adcdata_win, RangFFT).';
    fft_data(:, 1:4) = 0;  % 去直流

    % 逐帧取目标距离bin相位
    phase_seq = atan2(imag(fft_data(:, max_num)), real(fft_data(:, max_num)));

    % 相位解缠绕 + 差分 + 平滑
    phase_uw   = unwrap(phase_seq);
    phase_diff = [diff(phase_uw); phase_uw(end) - phase_uw(end-1)];
    phase_diff = movmean(phase_diff, 5);

    % 呼吸滤波 0.1–0.5 Hz
    [b_br, a_br] = butter(4, [0.1 0.5]/(fs/2), 'bandpass');
    breath_d = filtfilt(b_br, a_br, phase_diff);

    % 心跳滤波 1.0–2.0 Hz
    [b_hr, a_hr] = butter(4, [1.0 2.0]/(fs/2), 'bandpass');
    heart_d  = filtfilt(b_hr, a_hr, phase_diff);

    % 频率估计（在有效频段内搜峰）
    fshift = (-numFrames/2:numFrames/2-1) * (fs/numFrames);
    breath_fre = abs(fftshift(fft(breath_d)));
    heart_fre  = abs(fftshift(fft(heart_d)));
    mask_br = (fshift >= 0.1) & (fshift <= 0.5);
    mask_hr = (fshift >= 1.0) & (fshift <= 2.0);
    [~, bi] = max(breath_fre .* mask_br');
    [~, hi] = max(heart_fre  .* mask_hr');
    breath_counts(ai) = fshift(bi) * 60;
    heart_counts(ai)  = fshift(hi) * 60;

    breath_signals{ai} = breath_d;
    heart_signals{ai}  = heart_d;
    fprintf('[角度 %.1f°] 呼吸: %.1f 次/min，心跳: %.1f 次/min\n', ...
        theta_sel, breath_counts(ai), heart_counts(ai));

    % ===== 每个角度一张图，4行2列 =====
    fig_title = sprintf('角度 %.1f°', theta_sel);
    time_axis = (0:numFrames-1) * T_frame;
    range_axis_plot = (0:numADCSamples/2-1) * detaR;
    range_fft_1chirp = fft(process_adc(:,1), numADCSamples);
    phase_seq_raw = atan2(imag(fft_data(:, max_num)), real(fft_data(:, max_num)));

    figure('Color', 'w', 'Name', fig_title);
    sgtitle(fig_title, 'FontSize', 13, 'FontWeight', 'bold');

    subplot(4, 2, 1);
    plot(range_axis_plot, db(abs(range_fft_1chirp(1:numADCSamples/2))));
    xlabel('距离（m）'); ylabel('幅度(dB)');
    title('距离维FFT（第1帧）'); grid on;

    subplot(4, 2, 2);
    plot(phase_seq_raw);
    xlabel('帧序号'); ylabel('相位（rad）');
    title('原始相位'); grid on;

    subplot(4, 2, 3);
    plot(phase_uw);
    xlabel('帧序号'); ylabel('相位（rad）');
    title('解缠绕后相位'); grid on;

    subplot(4, 2, 4);
    plot(phase_diff);
    xlabel('帧序号'); ylabel('相位差分');
    title('相位差分'); grid on;

    subplot(4, 2, 5);
    plot(time_axis, breath_d, 'g');
    xlabel('时间（s）'); ylabel('幅度');
    title(sprintf('呼吸时域（%.1f 次/分钟）', breath_counts(ai))); grid on;

    subplot(4, 2, 6);
    plot(fshift, breath_fre);
    xlabel('频率（Hz）'); ylabel('幅度');
    title('呼吸FFT谱'); xlim([0 1]); grid on;

    subplot(4, 2, 7);
    plot(time_axis, heart_d, 'r');
    xlabel('时间（s）'); ylabel('幅度');
    title(sprintf('心跳时域（%.1f 次/分钟）', heart_counts(ai))); grid on;

    subplot(4, 2, 8);
    plot(fshift, heart_fre);
    xlabel('频率（Hz）'); ylabel('幅度');
    title('心跳FFT谱'); xlim([0 3]); grid on;
end

% ===== 汇总结果波形（所有角度合并一张图） =====
time_axis = (0:numFrames-1) * T_frame;
figure('Color', 'w', 'Name', '多角度生命体征检测结果');
for ai = 1:numAngles
    subplot(numAngles, 2, (ai-1)*2 + 1);
    plot(time_axis, breath_signals{ai}, 'g');
    grid on; xlabel('时间（s）'); ylabel('幅度');
    title(sprintf('呼吸 [%.1f°]: %.1f 次/分钟', selected_angles(ai), breath_counts(ai)));

    subplot(numAngles, 2, (ai-1)*2 + 2);
    plot(time_axis, heart_signals{ai}, 'r');
    grid on; xlabel('时间（s）'); ylabel('幅度');
    title(sprintf('心跳 [%.1f°]: %.1f 次/分钟', selected_angles(ai), heart_counts(ai)));
end
