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
chirpLoop = 2;      % numLoops=64, 2TX TDM -> 128 chirps/frame

%% 雷达参数设置
%Fs=4e6;             %ADC采样率 (原始IWR1642)
Fs=10e6;             %ADC采样率 (AWR1843)
c=3*1e8;            %光速
ts=numADCSamples/Fs;%ADC采样时间
slope=70.006e12;        %调频斜率 (原始IWR1642)
%slope=29.982e12;     %调频斜率 (AWR1843, 20260317数据配置)

B_valid =ts*slope;  %有效带宽
detaR=c/(2*B_valid);%距离分辨率

%% 读取Bin文件
%Filename = 'xxx.bin';  %文件名 (原始)
%Filename = 'E:\NJU\Beamforming_for_Sensing\data\mmwavestudio\Front_0.58_static_70.006_pause\adc_data.bin';  %AWR1843数据
Filename = 'E:\NJU\Beamforming_for_Sensing\data\mmwavestudio\20260326_1Person_1.7m_R15deg_Static\adc_data.bin';  %AWR1843数据
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

% 步骤4：构造 17° 方向的 CBF 导向矢量并波束成形
% 8元ULA，天线间距 d=λ/2：a(θ) = exp(-j*π*n*sin(θ))
theta_deg = -17;
theta_rad = theta_deg * pi / 180;
n = (0:7)';
steering_vec = exp(-1j * pi * n * sin(theta_rad));  % [8, 1]
w = steering_vec / norm(steering_vec);               % 归一化CBF权重

% 对每个 loop 做波束成形：y = w^H * x，输出 [numADCSamples, numLoops]
bf_output = zeros(numADCSamples, numLoops);
for loop = 1:numLoops
    bf_output(:, loop) = w' * va_data(:, :, loop);
end
fprintf('波束形成完成，指向角度: %d 度，虚拟阵列: 8元，输出维度: [%d, %d]\n', ...
    theta_deg, numADCSamples, numLoops);

% 步骤5：每帧取1个loop的数据用于生命体征检测（共1200帧）
% 原逻辑 1:128:end 是每帧取第1个TX0 chirp，这里对应每64个loop取1个
process_adc = bf_output(:, 1:chirpLoop:end);  % [numADCSamples, 1200]

%% 距离维FFT（1个chirp)
figure;
range_fft_1chirp = fft(process_adc(:,1), numADCSamples);
range_axis = (0:numADCSamples/2-1) * detaR;
plot(range_axis, db(abs(range_fft_1chirp(1:numADCSamples/2))));%取正频率半边
xlabel('距离（m）');
ylabel('幅度(dB)');
title('距离维FFT（1个chirp）');

%% 相位解缠绕部分
RangFFT = 512;
fft_data_last = zeros(1,RangFFT);  
range_max = 0;
adcdata = process_adc;
numChirps = size(adcdata, 2);%获取有多少个frame

%% 距离维FFT
% 在距离FFT前加 Hann 窗，降低距离旁瓣泄漏
hann_win = hann(numADCSamples);
adcdata_win = adcdata .* repmat(hann_win, 1, numChirps);
fft_data = fft(adcdata_win, RangFFT);
fft_data = fft_data.';
fft_data_abs = abs(fft_data);
fft_data_abs(:,1:4)=0; %去除直流分量
real_data = real(fft_data);
imag_data = imag(fft_data);


for i = 1:numChirps
    for j = 1:RangFFT  %对每一个距离点取相位 extract phase
        angle_fft(i,j) = atan2(imag_data(i, j),real_data(i, j));
    end
end

% =========================================================================
% 手动指定人体距离（比自动选能量最大点更可靠）
% 修改 target_range_m 为实际人体距离（单位：米）
target_range_m = 1.92;  % <-- 在这里修改目标距离

% 距离分辨率（基于 RangFFT 点数）
detaR_fft = c / (2 * slope * ts) / RangFFT * numADCSamples;
max_num = round(target_range_m / detaR_fft) + 1;
max_num = max(2, min(max_num, RangFFT));  % 防止越界，避开直流bin=1
fprintf('目标距离: %.2f m → 距离bin: %d（分辨率: %.4f m/bin）\n', ...
    target_range_m, max_num, detaR_fft);
% =========================================================================

%% 取出能量最大点的相位  extract phase from selected range bin
angle_fft_last = angle_fft(:,max_num);

%% 原始相位波形
figure;
plot(angle_fft_last);
xlabel('点数（N）');
ylabel('相位（rad）');
title('原始相位（选中距离bin）');

%% 进行相位解缠  phase unwrapping(手动解)，自动解可以采用MATLAB自带的函数unwrap()
n = 1;
for i = 1+1:numChirps
    diff = angle_fft_last(i) - angle_fft_last(i-1);
    if diff > pi
        angle_fft_last(i:end) = angle_fft_last(i:end) - 2*pi;
        n = n + 1;
    elseif diff < -pi
        angle_fft_last(i:end) = angle_fft_last(i:end) + 2*pi;  
    end
end

%% 解缠绕后相位波形
figure;
plot(angle_fft_last);
xlabel('点数（N）');
ylabel('相位（rad）');
title('解缠绕后相位（选中距离bin）');

%% phase difference 相位差分后的数据
angle_fft_last2=zeros(1,numChirps);
for i = 1:numChirps-1
    angle_fft_last2(i) = angle_fft_last(i+1) - angle_fft_last(i);
    angle_fft_last2(numChirps)=angle_fft_last(numChirps)-angle_fft_last(numChirps-1);
end 

figure;
plot(angle_fft_last2);
xlabel('点数（N）');
ylabel('相位');
title('相位差分后的结果');

%%  Butterworth带通滤波 + filtfilt零相位滤波
%fs =20; %呼吸心跳信号采样率 (原始IWR1642, 帧周期50ms)
fs =20; %呼吸心跳信号采样率 (AWR1843, 帧周期25ms → 1/0.025=40Hz)

% 呼吸滤波: 0.1-0.5 Hz
[b_breath, a_breath] = butter(4, [0.1 0.5]/(fs/2), 'bandpass');
breath_data = filtfilt(b_breath, a_breath, angle_fft_last2);

figure;
plot(breath_data);
xlabel('时间/点数');
ylabel('幅度');
title('呼吸时域波形');

%% 谱估计 -FFT -Peak interval
N1=length(breath_data);
fshift = (-N1/2:N1/2-1)*(fs/N1); % zero-centered frequency
breath_fre = abs(fftshift(fft(breath_data)));              %--FFT
figure;
plot(fshift,breath_fre);
xlabel('频率（f/Hz）');
ylabel('幅度');
title('呼吸信号FFT');

breath_fre_max = 0; % 呼吸频率

breath_index1 = length(breath_fre)/2;

for i = 1:breath_index1%谱峰最大值搜索 对称其实可以取一半
    if (breath_fre(i) > breath_fre_max)    
        breath_fre_max = breath_fre(i);
        if(breath_fre_max<1e-2) %幅度置信 判断是否是存在人的呼吸
            breath_index=numChirps+1;
        else
            breath_index=i;
        end
    end
end

breath_count =(fs*(breath_index1-breath_index)/numChirps)*60; %呼吸频率解算

%% 心跳滤波: 1-2Hz
[b_heart, a_heart] = butter(4, [1.0 2.0]/(fs/2), 'bandpass');
heart_data = filtfilt(b_heart, a_heart, angle_fft_last2);

N1=length(heart_data);
fshift = (-N1/2:N1/2-1)*(fs/N1); % zero-centered frequency
heart_fre = abs(fftshift(fft(heart_data))); 
figure;
plot(fshift,heart_fre);
xlabel('频率（f/Hz）');
ylabel('幅度');
title('心跳信号FFT');

heart_fre_max = 0; 
heart_index1 = length(heart_fre)/2 ;

for i = 1:heart_index1
    if (heart_fre(i) > heart_fre_max)    
        heart_fre_max = heart_fre(i);
        if(heart_fre_max<1e-2)%幅度置信 判断是否是存在人的心跳
            heart_index=numChirps+1;
        else
            heart_index=i;
        end
    end
end
heart_count =(fs*(heart_index1-(heart_index-1))/numChirps)*60%心跳频率解算

% 2399个帧，约为120s，
% 如果数据长度够长，则雷达会51.2s对呼吸数据和心跳数据进行一次刷新，
%以便实现更为精确的检测。

disp(['呼吸：',num2str(breath_count),'  心跳：',num2str(heart_count)])

%% 静态显示完整呼吸和心跳波形
%T_frame =0.05 ;%50ms (原始IWR1642)
T_frame =0.025 ;%25ms (AWR1843)
time_axis = (0:numChirps-1) * T_frame;

figure(10);
subplot(121);
plot(time_axis, breath_data, 'g');
grid on;
xlabel('时间（s）');
ylabel('幅度');
title(['呼吸时域波形：', num2str(breath_count, '%.1f'), ' 次/分钟']);

subplot(122);
plot(time_axis, heart_data, 'r');
grid on;
xlabel('时间（s）');
ylabel('幅度');
title(['心跳时域波形：', num2str(heart_count, '%.1f'), ' 次/分钟']);


