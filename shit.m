%% 功能：IWR1843EVM + DCA1000 ADC 数据处理
% 1) 解析二进制文件
% 2) 重组为 [4, ADCs, Chirps]
% 3) TDM-MIMO 虚拟阵列构建（8 元 ULA, d = λ/2）
% 4) 距离 FFT + MVDR/Capon 波束形成估角
% 5) 交互式多角度选择
% 6) CBF 波束形成提取相位序列
% 7) 生命体征处理：Hann窗、去直流、相位解缠绕、位移化、平滑、带通FFT估频
% 8) MSense 运动干扰消除：对每个 target 遍历所有 reference
%
% 说明：
% - 不修改角度选择逻辑
% - 不再只使用相邻角作为 reference
% - 对每个 target，遍历所有其他已选角度作为 reference
% - 相位链路采用更贴近论文的方式：复数除法 + angle + unwrap + 位移恢复
% - 不使用 diff

clc; clear; close all;

%% =========================================================================
%% 读取数据部分
numADCSamples = 256;   % number of ADC samples per chirp
numADCBits    = 16;    % number of ADC bits per sample
numRX         = 4;     % number of receivers
numLanes      = 2;     % do not change
isReal        = 0;     % 1: real-only data, 0: complex data

chirpLoop = 2;
%% =========================================================================

%% 雷达参数设置
Fs    = 10e6;          % ADC采样率
c     = 3e8;           % 光速
ts    = numADCSamples / Fs;

slope = 70.006e12;     % 调频斜率
B_valid = ts * slope;
detaR   = c / (2 * B_valid);

fc     = 77e9;
lambda = c / fc;
d      = lambda / 2;

%% 读取Bin文件
Filename = 'E:\NJU\Beamforming_for_Sensing\data\mmwavestudio\20260406_1Person_0.6m_0deg_Hotting\adc_data.bin';

fid = fopen(Filename, 'r');
if fid < 0
    error('无法打开文件：%s', Filename);
end

adcDataRow = fread(fid, 'int16');

if numADCBits ~= 16
    l_max = 2^(numADCBits-1) - 1;
    adcDataRow(adcDataRow > l_max) = adcDataRow(adcDataRow > l_max) - 2^numADCBits;
end
fclose(fid);

fileSize = size(adcDataRow, 1);
PRTnum = fix(fileSize / (numADCSamples * numRX));
fileSize = PRTnum * numADCSamples * numRX;
adcData = adcDataRow(1:fileSize);

if isReal
    numChirps = fileSize / numADCSamples / numRX;
    LVDS = reshape(adcData, numADCSamples * numRX, numChirps).';
else
    numChirps = fileSize / 2 / numADCSamples / numRX;
    LVDS = zeros(1, fileSize / 2);

    counter = 1;
    for i = 1:4:fileSize-1
        LVDS(1, counter)   = adcData(i)   + 1j * adcData(i+2);
        LVDS(1, counter+1) = adcData(i+1) + 1j * adcData(i+3);
        counter = counter + 2;
    end
    LVDS = reshape(LVDS, numADCSamples * numRX, numChirps).';
end

%% 重组数据
adcData = zeros(numRX, numChirps * numADCSamples);
for row = 1:numRX
    for i = 1:numChirps
        adcData(row, (i-1)*numADCSamples+1:i*numADCSamples) = ...
            LVDS(i, (row-1)*numADCSamples+1:row*numADCSamples);
    end
end

%% =========================================================================
%% TDM-MIMO 虚拟阵列构建
rx_all = zeros(numRX, numADCSamples, numChirps);
for rx = 1:numRX
    rx_all(rx, :, :) = reshape(adcData(rx, :), numADCSamples, numChirps);
end

tx0_data = rx_all(:, :, 1:2:end);
tx2_data = rx_all(:, :, 2:2:end);
numLoops = size(tx0_data, 3);

va_data = zeros(8, numADCSamples, numLoops);
for loop = 1:numLoops
    tx0_rev = tx0_data(end:-1:1, :, loop);
    tx2_rev = tx2_data(end:-1:1, :, loop);
    va_data(:, :, loop) = [tx0_rev; tx2_rev];
end

clear adcDataRow adcData LVDS rx_all tx0_data tx2_data

%% =========================================================================
%% 第一帧 MVDR Range-Angle 图
ra_ang_scan = -60:0.5:60;
RangFFT_ra = 512;
diag_load_alpha_ra = 0.1;

loops_per_frame = 2;
frame1_idx = 1:min(loops_per_frame, numLoops);

detaR_fft_ra = c / (2 * slope * ts) / RangFFT_ra * numADCSamples;
range_axis_ra = (0:RangFFT_ra-1) * detaR_fft_ra;

hann_win_ra = hann(numADCSamples);
Xr_all = zeros(RangFFT_ra, 8, numel(frame1_idx));

for ii = 1:numel(frame1_idx)
    loop_id = frame1_idx(ii);
    x_in = va_data(:, :, loop_id).';
    x_win = x_in .* repmat(hann_win_ra, 1, 8);
    Xr_all(:, :, ii) = fft(x_win, RangFFT_ra, 1);
end

Xr_mean = mean(Xr_all, 3);
Xr_all = Xr_all - repmat(Xr_mean, 1, 1, size(Xr_all,3));

n_ra = (0:7).';
RA_mvdr = zeros(RangFFT_ra, numel(ra_ang_scan));

for rb = 2:RangFFT_ra
    Rxx_sum_ra = zeros(8, 8);
    for ii = 1:size(Xr_all,3)
        x_bin = squeeze(Xr_all(rb, :, ii)).';
        Rxx_sum_ra = Rxx_sum_ra + (x_bin * x_bin');
    end
    Rxx_ra = Rxx_sum_ra / size(Xr_all,3);

    diag_load_ra = diag_load_alpha_ra * real(trace(Rxx_ra)) / 8;
    Rxx_loaded_ra = Rxx_ra + diag_load_ra * eye(8);

    for k = 1:numel(ra_ang_scan)
        a_ra = exp(-1j * pi * n_ra * sin(ra_ang_scan(k) * pi / 180));
        RA_mvdr(rb, k) = 1 / real(a_ra' * (Rxx_loaded_ra \ a_ra) + eps);
    end
end

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

%% =========================================================================
%% MVDR 估角
n = (0:7).';

target_range_m_mvdr = 0.58;
RangFFT_bf = 512;
ang_scan = -60:0.2:60;
diag_load_alpha = 0.01;

detaR_fft_bf = c / (2 * slope * ts) / RangFFT_bf * numADCSamples;
target_bin_mvdr = round(target_range_m_mvdr / detaR_fft_bf) + 1;
target_bin_mvdr = max(2, min(target_bin_mvdr, RangFFT_bf));

hann_win_bf = hann(numADCSamples);
Rxx_sum = zeros(8, 8);

for loop = 1:numLoops
    x_in = va_data(:, :, loop).';
    x_win = x_in .* repmat(hann_win_bf, 1, 8);
    Xr = fft(x_win, RangFFT_bf, 1);
    x_bin = Xr(target_bin_mvdr, :).';
    Rxx_sum = Rxx_sum + (x_bin * x_bin');
end

Rxx = Rxx_sum / max(numLoops, 1);

diag_load = diag_load_alpha * real(trace(Rxx)) / 8;
Rxx_loaded = Rxx + diag_load * eye(8);

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

P_mvdr_db = 10 * log10(P_mvdr / max(P_mvdr) + eps);

figure('Color','w');
plot(ang_scan, P_mvdr_db, 'm', 'LineWidth', 1.8); hold on;
xline(theta_mvdr_deg, '--k', sprintf('\\theta_{MVDR}=%.1f°', theta_mvdr_deg), 'LineWidth', 1.2);
grid on;
xlabel('角度（°）');
ylabel('归一化谱功率（dB）');
title('固定距离bin的MVDR角谱 — 左键点击选角度，右键结束');
xlim([ang_scan(1) ang_scan(end)]);
ylim([-40 5]);

%% =========================================================================
%% 交互式多角度选择
fprintf('请在MVDR角谱图上左键点击选择目标角度，右键结束...\n');
selected_angles = [];
while true
    [x_click, ~, button] = ginput(1);
    if button == 3
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

%% =========================================================================
%% 多角度 CBF + 生命体征提取
numAngles = numel(selected_angles);
RangFFT = 512;
hann_win = hann(numADCSamples);
fs = 20;
T_frame = 0.05;

target_range_m = 0.58;
detaR_fft = c / (2 * slope * ts) / RangFFT * numADCSamples;
max_num = round(target_range_m / detaR_fft) + 1;
max_num = max(2, min(max_num, RangFFT));

fprintf('目标距离: %.2f m → 距离bin: %d（分辨率: %.4f m/bin）\n', ...
    target_range_m, max_num, detaR_fft);

breath_counts  = zeros(1, numAngles);
heart_counts   = zeros(1, numAngles);
breath_signals = cell(1, numAngles);
heart_signals  = cell(1, numAngles);

angle_y         = cell(1, numAngles);
angle_phase_raw = cell(1, numAngles);
angle_phase_uw  = cell(1, numAngles);
angle_disp_mm   = cell(1, numAngles);

for ai = 1:numAngles
    theta_sel = selected_angles(ai);
    theta_rad = theta_sel * pi / 180;
    steering_vec = exp(-1j * pi * n * sin(theta_rad));
    w = steering_vec / norm(steering_vec);

    bf_output = zeros(numADCSamples, numLoops);
    for loop = 1:numLoops
        bf_output(:, loop) = (w' * va_data(:, :, loop)).';
    end

    process_adc = bf_output(:, 1:chirpLoop:end);
    numFrames = size(process_adc, 2);

    adcdata_win = process_adc .* repmat(hann_win, 1, numFrames);
    fft_data = fft(adcdata_win, RangFFT).';
    fft_data(:, 1:4) = 0;

    angle_y{ai} = fft_data(:, max_num);

    phase_seq_raw = angle(fft_data(:, max_num));
    phase_uw = unwrap(phase_seq_raw);

    disp_seq_mm = phase_uw * lambda / (4*pi) * 1000;
    disp_seq_mm = detrend(disp_seq_mm);
    disp_seq_mm = movmean(disp_seq_mm, 5);

    angle_phase_raw{ai} = phase_seq_raw;
    angle_phase_uw{ai}  = phase_uw;
    angle_disp_mm{ai}   = disp_seq_mm;

    [b_br, a_br] = butter(4, [0.1 0.5] / (fs/2), 'bandpass');
    breath_d = filtfilt(b_br, a_br, disp_seq_mm);

    [b_hr, a_hr] = butter(4, [1.0 2.0] / (fs/2), 'bandpass');
    heart_d = filtfilt(b_hr, a_hr, disp_seq_mm);

    N_fft_eval = max(1024, 2^nextpow2(numFrames));
    breath_fre = abs(fft(breath_d, N_fft_eval));
    heart_fre  = abs(fft(heart_d,  N_fft_eval));
    f_axis_eval = (0:N_fft_eval-1) * (fs / N_fft_eval);

    mask_br = (f_axis_eval >= 0.1) & (f_axis_eval <= 0.5);
    mask_hr = (f_axis_eval >= 1.0) & (f_axis_eval <= 2.0);

    tmp_br = breath_fre;
    tmp_br(~mask_br) = 0;
    [~, bi] = max(tmp_br);
    breath_counts(ai) = f_axis_eval(bi) * 60;

    tmp_hr = heart_fre;
    tmp_hr(~mask_hr) = 0;
    [~, hi] = max(tmp_hr);
    heart_counts(ai) = f_axis_eval(hi) * 60;

    breath_signals{ai} = breath_d;
    heart_signals{ai}  = heart_d;

    fprintf('[角度 %.1f°] 呼吸: %.1f 次/min，心跳: %.1f 次/min\n', ...
        theta_sel, breath_counts(ai), heart_counts(ai));

    fig_title = sprintf('角度 %.1f°', theta_sel);
    time_axis = (0:numFrames-1) * T_frame;
    range_axis_plot = (0:numADCSamples/2-1) * detaR;
    range_fft_1chirp = fft(process_adc(:,1), numADCSamples);

    figure('Color', 'w', 'Name', fig_title);
    sgtitle(fig_title, 'FontSize', 13, 'FontWeight', 'bold');

    subplot(4, 2, 1);
    plot(range_axis_plot, 20*log10(abs(range_fft_1chirp(1:numADCSamples/2)) + eps));
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
    plot(disp_seq_mm);
    xlabel('帧序号'); ylabel('位移（mm）');
    title('相位换算位移后序列'); grid on;

    subplot(4, 2, 5);
    plot(time_axis, breath_d, 'g');
    xlabel('时间（s）'); ylabel('幅度');
    title(sprintf('呼吸时域（%.1f 次/分钟）', breath_counts(ai)));
    grid on;

    subplot(4, 2, 6);
    plot(f_axis_eval, breath_fre);
    xlabel('频率（Hz）'); ylabel('幅度');
    title('呼吸FFT谱');
    xlim([0 1]); grid on;

    subplot(4, 2, 7);
    plot(time_axis, heart_d, 'r');
    xlabel('时间（s）'); ylabel('幅度');
    title(sprintf('心跳时域（%.1f 次/分钟）', heart_counts(ai)));
    grid on;

    subplot(4, 2, 8);
    plot(f_axis_eval, heart_fre);
    xlabel('频率（Hz）'); ylabel('幅度');
    title('心跳FFT谱');
    xlim([0 3]); grid on;
end

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

%% =========================================================================
%% MSense：对每个 target 遍历所有 reference
if numAngles < 2
    fprintf('[MSense] 选中角度少于2个，无法进行遍历参考角干扰消除。\n');
else
    fprintf('\n[MSense] 开始遍历所有 reference 角进行干扰消除...\n');

    msense_clean_disp   = cell(numAngles, numAngles);
    msense_clean_breath = cell(numAngles, numAngles);
    msense_clean_heart  = cell(numAngles, numAngles);
    msense_br_bpm       = nan(numAngles, numAngles);
    msense_hr_bpm       = nan(numAngles, numAngles);

    N_fft_eval_ms = 2048;
    f_axis_eval_ms = (0:N_fft_eval_ms-1) * (fs / N_fft_eval_ms);

    [b_br_ms, a_br_ms] = butter(4, [0.1 0.5] / (fs/2), 'bandpass');
    [b_hr_ms, a_hr_ms] = butter(4, [1.0 2.0] / (fs/2), 'bandpass');

    for ai = 1:numAngles
        theta_tar_deg = selected_angles(ai);
        fprintf('\n[MSense] ===== Target %.1f° =====\n', theta_tar_deg);

        for ri = 1:numAngles
            if ri == ai
                continue;
            end

            theta_ref_deg = selected_angles(ri);
            fprintf('[MSense] target=%.1f° -> ref=%.1f°\n', theta_tar_deg, theta_ref_deg);

            y_tar = angle_y{ai};
            y_ref = angle_y{ri};

            L = min(length(y_tar), length(y_ref));
            y_tar_use = y_tar(1:L);
            y_ref_use = y_ref(1:L);

            % 论文更贴近的复数除法
            y_new = y_tar_use .* conj(y_ref_use) ./ (abs(y_ref_use).^2 + eps);

            % angle -> unwrap -> 位移
            phase_new = angle(y_new);
            phase_uw_new = unwrap(phase_new);

            disp_new_mm = phase_uw_new * lambda / (4*pi) * 1000;
            disp_new_mm = detrend(disp_new_mm);
            disp_new_mm = movmean(disp_new_mm, 5);

            % 带通滤波
            breath_sig = filtfilt(b_br_ms, a_br_ms, disp_new_mm);
            heart_sig  = filtfilt(b_hr_ms, a_hr_ms, disp_new_mm);

            % 频率估计
            spec_br = abs(fft(breath_sig, N_fft_eval_ms));
            spec_hr = abs(fft(heart_sig,  N_fft_eval_ms));

            mask_br = (f_axis_eval_ms >= 0.1) & (f_axis_eval_ms <= 0.5);
            mask_hr = (f_axis_eval_ms >= 1.0) & (f_axis_eval_ms <= 2.0);

            tmp_br = spec_br;
            tmp_br(~mask_br) = 0;
            [~, idx_br] = max(tmp_br);
            br_hz = f_axis_eval_ms(idx_br);
            br_bpm = br_hz * 60;

            tmp_hr = spec_hr;
            tmp_hr(~mask_hr) = 0;
            [~, idx_hr] = max(tmp_hr);
            hr_hz = f_axis_eval_ms(idx_hr);
            hr_bpm = hr_hz * 60;

            msense_clean_disp{ai, ri}   = disp_new_mm;
            msense_clean_breath{ai, ri} = breath_sig;
            msense_clean_heart{ai, ri}  = heart_sig;
            msense_br_bpm(ai, ri)       = br_bpm;
            msense_hr_bpm(ai, ri)       = hr_bpm;

            fprintf('[MSense] (target %.1f°, ref %.1f°) -> 呼吸 %.1f 次/min，心跳 %.1f 次/min\n', ...
                theta_tar_deg, theta_ref_deg, br_bpm, hr_bpm);

            % 单独画每一对 target-ref 的结果
            time_ms = (0:L-1) * T_frame;

            figure('Color', 'w', 'Name', sprintf('MSense_T%.1f_R%.1f', theta_tar_deg, theta_ref_deg));

            subplot(4,1,1);
            plot(time_ms, angle_disp_mm{ai}(1:L), 'b');
            grid on;
            xlabel('时间（s）'); ylabel('位移（mm）');
            title(sprintf('目标角 %.1f° 原始位移', theta_tar_deg));

            subplot(4,1,2);
            plot(time_ms, angle_disp_mm{ri}(1:L), 'm');
            grid on;
            xlabel('时间（s）'); ylabel('位移（mm）');
            title(sprintf('参考角 %.1f° 原始位移', theta_ref_deg));

            subplot(4,1,3);
            plot(time_ms, disp_new_mm, 'k');
            grid on;
            xlabel('时间（s）'); ylabel('位移（mm）');
            title(sprintf('干扰消除后位移：target %.1f° / ref %.1f°', theta_tar_deg, theta_ref_deg));

            subplot(4,1,4);
            plot(time_ms, breath_sig, 'g', 'LineWidth', 1.2); hold on;
            plot(time_ms, heart_sig, 'r', 'LineWidth', 1.0);
            grid on;
            xlabel('时间（s）'); ylabel('幅度');
            title(sprintf('呼吸 %.1f bpm，心跳 %.1f bpm', br_bpm, hr_bpm));
            legend('呼吸', '心跳');

            sgtitle(sprintf('MSense逐对遍历：Target %.1f° / Reference %.1f°', ...
                theta_tar_deg, theta_ref_deg));
        end
    end

    %% 汇总表格输出
    fprintf('\n================ MSense 呼吸频率矩阵（bpm） ================\n');
    fprintf('行 = target angle，列 = reference angle；对角线无效\n');
    disp(array2table(msense_br_bpm, ...
        'VariableNames', matlab.lang.makeValidName(compose('Ref_%.1fdeg', selected_angles)), ...
        'RowNames', matlab.lang.makeValidName(compose('Tar_%.1fdeg', selected_angles))));

    fprintf('\n================ MSense 心跳频率矩阵（bpm） ================\n');
    fprintf('行 = target angle，列 = reference angle；对角线无效\n');
    disp(array2table(msense_hr_bpm, ...
        'VariableNames', matlab.lang.makeValidName(compose('Ref_%.1fdeg', selected_angles)), ...
        'RowNames', matlab.lang.makeValidName(compose('Tar_%.1fdeg', selected_angles))));

    %% 呼吸频率热图
    figure('Color', 'w', 'Name', 'MSense 呼吸频率矩阵');
    imagesc(msense_br_bpm);
    colorbar;
    xlabel('Reference Angle Index');
    ylabel('Target Angle Index');
    title('MSense 呼吸频率矩阵（bpm）');
    xticks(1:numAngles); yticks(1:numAngles);
    xticklabels(compose('%.1f°', selected_angles));
    yticklabels(compose('%.1f°', selected_angles));
    axis equal tight;

    %% 心跳频率热图
    figure('Color', 'w', 'Name', 'MSense 心跳频率矩阵');
    imagesc(msense_hr_bpm);
    colorbar;
    xlabel('Reference Angle Index');
    ylabel('Target Angle Index');
    title('MSense 心跳频率矩阵（bpm）');
    xticks(1:numAngles); yticks(1:numAngles);
    xticklabels(compose('%.1f°', selected_angles));
    yticklabels(compose('%.1f°', selected_angles));
    axis equal tight;
end
