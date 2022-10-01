% 交替方向乘子法

rng('default') % For reproducibility

% 考虑一个20 节点的分布式系统。节点i 有线性测量bi=Aix+ei，其中bi为10 维的测量
% 值，Ai为10ⅹ300 维的测量矩阵，x 为300 维的未知稀疏向量且稀疏度为5，ei为10 维的测
% 量噪声。从所有bi与Ai中恢复x 的一范数规范化最小二乘模型如下：
% min (1/2)||A1x-b1||2
% 2+…+(1/2)||A10x-b10||2
% 2+p||x||1
% 其中p 为非负的正则化参数。请设计下述分布式算法求解该问题：
% 1、邻近点梯度法；
% 2、交替方向乘子法；
% 3、次梯度法；
% 在实验中，设x 的真值中的非零元素服从均值为0 方差为1 的高斯分布，Ai中的元素服从均
% 值为0 方差为1 的高斯分布，ei中的元素服从均值为0 方差为0.2 的高斯分布。对于每种算
% 法，请给出每步计算结果与真值的距离以及每步计算结果与最优解的距离。此外，请讨论正
% 则化参数p 对计算结果的影响。

x_true = sprandn(300,1,5/300);
x_true = full(x_true);

A = normrnd(0,1,10,300,20);
e = normrnd(0,0.2,10,1,20);
b = pagemtimes(A, x_true)+e;

u = zeros(300, 20);
x = zeros(300, 20);
z = zeros(300, 1);

rou = 0.001;

% change this
p = 0.02;

iter = 10000;
x_mean = zeros(300, 1);
x_mean_n = zeros(300, iter);
Inv = zeros(300, 300, 20);
for i = 1:20
    Inv(:, :, i) = inv(A(:, :, i)'*A(:, :, i) - rou*eye(300));
end

for i = 1:iter
    [u, x] = arrayfun(@(i) update_u_x(A(:, :, i), b(:, i), u(:, i), x(:, i), Inv(:, :, i), ...
            z, rou), 1:20, 'UniformOutput', false);
    u = cell2mat(u);
    x = cell2mat(x);
    u_mean = mean(u, 2);
    x_mean = mean(x, 2);
    z = sign(u_mean+x_mean).*max(abs(u_mean+x_mean)-p/(rou*20),0);
    x_mean_n(:, i) = x_mean;
end

distance_to_true = arrayfun(@(j) norm(x_mean_n(:, j)-x_true), 1:i);
[distance, m] = min(distance_to_true);
distance_to_optimal = arrayfun(@(j) norm(x_mean_n(:, j)-x_mean_n(:, m)), 1:i);

function [u, x] = update_u_x(A, b, u, x, Inv, z, rou)
    u = u + x - z;
    x = Inv*(A'*b - rou*(z-u));
end