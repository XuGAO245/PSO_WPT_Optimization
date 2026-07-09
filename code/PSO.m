% =========================================================================
% 第一部分：PSO主循环函数
% =========================================================================
% 初始化工作环境
clc;
clear;
close all hidden;
format long;

%% 模型与评价参数
mdl = 'WPT';
Vref=1;
Ns=1200;
PCTvst=0.01;

%% PSO 参数设置
N = 10;                         % 粒子数
Gmax = 100;                     % 最大迭代次数
plimit = [0.01, 1;              % kp 取值范围
          0.01, 1];             % ki 取值范围
D = size(plimit, 1);            % 变量维数
c1 = 2;                         % 个体学习因子
c2 = 2;                         % 群体学习因子
wMax = 0.9;                     % 最大惯性权重
wMin = 0.4;                     % 最小惯性权重
run_times = 1;                  % 独立运行次数

rand('state', sum(100 * clock));

%% 开始优化
for kk = 1:run_times
    disp(['runtimes = ', num2str(kk)]);
    disp(['generation = ', num2str(1)]);

    %% 粒子初始化
    x = zeros(N, D);            % 粒子位置
    v = zeros(N, D);            % 粒子速度
    fit = zeros(N, 1);          % 当前适应度
    PBest = zeros(N, D);        % 个体历史最优位置
    PBestVal = -inf(N, 1);      % 个体历史最优适应度
    GBestVal = -inf;            % 全局最优适应度

    % 随机初始化粒子位置
    for ii = 1:D
        x(:, ii) = plimit(ii, 1) + (plimit(ii, 2) - plimit(ii, 1)) * rand(N, 1);
    end

    % 随机初始化粒子速度
    v = rand(N, D);

    % 个体历史最优初值设为当前位置
    PBest = x;

    %% 计算初始粒子的适应度
    for ii = 1:N
        kp = x(ii, 1);
        ki = x(ii, 2);

        set_param([mdl '/Kp'], 'Value', num2str(kp));
        set_param([mdl '/Ki'], 'Value', num2str(ki));

        disp(['NO. ', num2str(ii)]);
        disp(['[kp, ki]=[', num2str(kp), ', ', num2str(ki), ']']);

        sim(mdl);

        t = Vout.time(:, 1);
        vol = Vout.signals.values(:, 1);
        fit(ii) = evaluation(Vref, PCTvst, Ns, t, vol);

        disp('=================================================================================');
    end

    %% 初始化个体最优和全局最优
    PBestVal = fit;

    if GBestVal < max(PBestVal)
        [GBestVal, index] = max(PBestVal);
        GBest = PBest(index, :);
    end

    disp('GBest_Temp:');
    disp(['[kp, ki]=[', num2str(GBest(1)), ' ,', num2str(GBest(2)), ']']);
    disp(['GBestVal=', num2str(GBestVal)]);
    disp('=================================================================================');

    %% PSO 主迭代
    for ger = 2:Gmax
        disp(['generation = ', num2str(ger)]);

        % 更新惯性权重
        w = Get_Omega(wMax, wMin, ger, Gmax);

        % 更新速度和位置
        for ii = 1:D
            v(:, ii) = w * v(:, ii) ...
                     + c1 * rand * (PBest(:, ii) - x(:, ii)) ...
                     + c2 * rand * (repmat(GBest(:, ii), N, 1) - x(:, ii));

            x(:, ii) = x(:, ii) + v(:, ii);

            % 越界处理
            if rand < 0.5
                x(x(:, ii) > plimit(ii, 2), ii) = plimit(ii, 2);
                x(x(:, ii) < plimit(ii, 1), ii) = plimit(ii, 1);
            else
                x(x(:, ii) > plimit(ii, 2), ii) = ...
                    plimit(ii, 1) + (plimit(ii, 2) - plimit(ii, 1)) * rand(sum((x(:, ii) > plimit(ii, 2)) ~= 0), 1);
                x(x(:, ii) < plimit(ii, 1), ii) = ...
                    plimit(ii, 1) + (plimit(ii, 2) - plimit(ii, 1)) * rand(sum((x(:, ii) < plimit(ii, 1)) ~= 0), 1);
            end
        end

        %% 计算新一代粒子的适应度，并更新个体最优
        for ii = 1:N
            kp = x(ii, 1);
            ki = x(ii, 2);

            set_param([mdl '/Kp'], 'Value', num2str(kp));
            set_param([mdl '/Ki'], 'Value', num2str(ki));

            disp(['NO. ', num2str(ii)]);
            disp(['[kp, ki]=[', num2str(kp), ', ' num2str(ki), ']']);

            sim(mdl);

            t = Vout.time(:, 1);
            vol = Vout.signals.values(:, 1);
            fit(ii) = evaluation(Vref, PCTvst, Ns, t, vol);

            disp('=================================================================================');

            % 更新个体历史最优
            if PBestVal(ii) < fit(ii)
                PBestVal(ii) = fit(ii);
                PBest(ii, :) = x(ii, :);
            end
        end

        %% 更新全局最优
        if GBestVal < max(PBestVal)
            [GBestVal, index] = max(PBestVal);
            GBest = PBest(index, :);
        end

        disp('GBest_Temp:');
        disp(['[kp, ki]=[', num2str(GBest(1)), ' ,', num2str(GBest(2)), ']']);
        disp(['GBestVal=', num2str(GBestVal)]);
        disp('=================================================================================');
        disp(' ');
    end
end

%% 输出最终最优结果并自动绘制对比图
disp(['[kp, ki]=[', num2str(GBest(1)), ' ,', num2str(GBest(2)), ']']);

disp('正在自动运行对照仿真并生成优化前后对比图...');

% 1. 配置传统经验控制参数并运行仿真，保存数据
kp_old = 1.0; 
ki_old = 1.0; % 对照组参数可根据实际情况修改
set_param([mdl '/Kp'], 'Value', num2str(kp_old));
set_param([mdl '/Ki'], 'Value', num2str(ki_old));
sim(mdl);
t_old = Vout.time(:, 1);
vol_old = Vout.signals.values(:, 1);

% 2. 重新配置优化后的最佳参数运行仿真，保存数据
set_param([mdl '/Kp'], 'Value', num2str(GBest(1)));
set_param([mdl '/Ki'], 'Value', num2str(GBest(2)));
sim(mdl);
t_new = Vout.time(:, 1);
vol_new = Vout.signals.values(:, 1);

% 3. 自动绘制响应曲线对比图
figure('Color', [1 1 1]);           
hold on;

% 绘制参考电压基准线
yline(Vref, 'm--', 'LineWidth', 1.2);

% 绘制双组对比波形
plot(t_new, vol_new, 'b-', 'LineWidth', 2.0);   % 优化后：蓝色加粗
plot(t_old, vol_old, 'r-', 'LineWidth', 1.5);   % 优化前：红色

% 坐标轴与网格精美格式配置
xlim([0 0.05]);
ylim([0 1.4]);
box on;                             
grid on;                            
ax = gca;
ax.GridLineStyle = ':';            
ax.GridAlpha = 0.5;                

% 轴标签与标题
xlabel('时间 (s)', 'FontName', '宋体', 'FontSize', 11);
ylabel('输出电压 (V)', 'FontName', '宋体', 'FontSize', 11);
title('WPT系统输出电压对比图 (PSO优化前后)', 'FontName', '宋体', 'FontSize', 13, 'FontWeight', 'bold');

% 动态组装图例参数
lgd_str1 = sprintf('优化后：[K_p, K_i] = [%.3f, %.3f]', GBest(1), GBest(2));
lgd_str2 = sprintf('优化前：[K_p, K_i] = [%.1f, %.1f]', kp_old, ki_old);
legend('参考电压', lgd_str1, lgd_str2, 'FontName', '宋体', 'FontSize', 10, 'Location', 'southeast');

hold off;
disp('性能对比图生成完毕！');


% =========================================================================
% 第二部分：局部函数区
% =========================================================================

%% 1. evaluation 函数 
function [fit] = evaluation(Vref, PCTvst, Ns, t, vol)
% 适应度函数
% 输入：
%   Vref   - 参考电压
%   PCTvst - 稳态误差允许百分比
%   Ns     - 用于稳态误差计算的采样点数
%   t      - 仿真时间序列
%   vol    - 输出电压序列
% 输出：
%   fit    - 适应度值

len = size(t, 1);   % 数据长度

%% ====================== 1. 调节时间 ST ======================
% 定义误差带
ub = Vref + Vref * PCTvst;
lb = Vref - Vref * PCTvst;

flag = 1;           
ST = t(len);        

for ii = 1:len
    % 首次进入稳态区间
    if flag && vol(ii) <= ub && vol(ii) >= lb
        ST = t(ii);     % 记录进入稳态的时间
        flag = 0;
    % 一旦离开稳态区间，重新判定
    elseif vol(ii) >= ub || vol(ii) <= lb
        ST = t(len);    % 认为尚未稳定
        flag = 1;
    end
end

%% ====================== 2. 超调量 OV ======================
delta = max(vol) - Vref;   % 峰值偏差

if delta > 0
    OV = delta;            % 有超调
else
    OV = 0;                % 无超调
end

%% ====================== 3. 稳态误差 SSE ======================
% 使用最后 Ns 个采样点计算稳态误差平方和
SSE = sum((vol(len-Ns+1:len, 1) - Vref).^2);

%% ====================== 4. 加权评价函数 ======================
% 各指标加权（权重可根据需求调整）
f1 = 2500 * ST;       % 调节时间权重
f2 = 1600 * OV;       % 超调权重
f3 = 1000 * SSE;      % 稳态误差权重

% 计算损耗
cost = f1 + f2 + f3;

% 计算适应度
fit = -1 * cost;

if isnan(fit)
    fit = 1;
end

%% ====================== 5. 输出调试信息 ======================
disp(['[ST, OV, SSE, f1, f2, f3, fit, cost]=[', ...
      num2str(ST), ', ', ...
      num2str(OV), ', ', ...
      num2str(SSE), ', ', ...
      num2str(f1), ', ', ...
      num2str(f2), ', ', ...
      num2str(f3), ', ', ...
      num2str(fit), ', ', ...
      num2str(cost), ']']);
end

%% 2. Get_Omega 函数
function w=Get_Omega(wMax, wMin, ger, Gmax)
w=wMax-ger*(wMax-wMin)/Gmax;
end