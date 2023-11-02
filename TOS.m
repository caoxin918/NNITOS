function[z_k] = TOS(surf_Nodes,surf_density,A_temp1, A_temp2, A)
warning('off');
% 计算拉普拉斯矩阵L = D - W, D是入度矩阵， W是高斯权重矩阵
surf_density_temp = surf_density;

NEAR_NODES = 9;
lambda_1 = 1e-5;
lambda_2 = 1e-5;
R_sigma = 0.4;
surf_NumNodes = size(surf_density,1);
NumNodes = size(A,2);

% 每个点作为中心顶点，给每个中心顶点找k个最近的结点，将k近邻结点的能量值保存在k_nearest_density中
% 将k近邻结点的欧几里得距离保存在k_nearest_distance，这个矩阵最后会被赋值给高斯加权矩阵L
k_nearest_density = zeros(surf_NumNodes, NEAR_NODES);
k_nearest_distance = zeros(surf_NumNodes, NEAR_NODES);
k_nearest_index = zeros(surf_NumNodes, NEAR_NODES);
L_W = zeros(NumNodes, NumNodes);
L_D = zeros(NumNodes, NumNodes);

% 给每个中心顶点找k个最近的结点
for i = 1:surf_NumNodes
    [Idx,Distance] = knnsearch(surf_Nodes, surf_Nodes(i,:),'K',9);
    k_nearest_index(i,:) = Idx;
    k_nearest_density(i,:) = surf_density_temp(Idx);
    k_nearest_distance(i,:) = Distance;
end

% k_nearest_density每一行的能量值减去每一行的均值，小于均值的点会被置0，大于该均值的点被置为1
k_nearest_density = k_nearest_density - mean(k_nearest_density,2);
k_nearest_density(find(k_nearest_density < 0)) = 0;
k_nearest_density(find(k_nearest_density > 0)) = 1;

% 用k_nearest_density点乘k_nearest_distance，小于均值的点高斯权重为0
k_nearest_distance = k_nearest_density .* k_nearest_distance;
% 高斯权重
k_nearest_distance = k_nearest_distance  / (-4 * R_sigma^2);
k_nearest_distance = exp(k_nearest_distance);
k_nearest_distance(find(k_nearest_distance == 1)) = 0;

% 将每一个元素的高斯权重都除以k近邻权重之和
k_nearest_distance = k_nearest_distance ./ sum(k_nearest_distance,2);
k_nearest_distance(find(isnan(k_nearest_distance))) = 0;

% 得到有向图高斯权重矩阵
for i=1:surf_NumNodes
    L_W(i,k_nearest_index(i,:)) = k_nearest_distance(i,:);
    L_D(i, i) = sum(L_W(i,:));
end

L = L_D - L_W;
I = eye(NumNodes);
gamma = 0.15;
y_k = zeros(NumNodes,1);

E_L2_sum = zeros(NumNodes,1);
E_cos_sum = zeros(NumNodes,1);
% 在ROI区域内点的下标
ROI_nodes = 1:1:surf_NumNodes;
ROI_nodes = ROI_nodes';
ROI_nodes = [ROI_nodes ,surf_Nodes];
outIdx_ROI = [];
% 记录迭代的次数
iter_i = 1;
inf_p = 0.1;

E_L2_0 = 0;
E_cos_0 = 0;
% 加入Region of Interest（ROI）
while 1
    
    z_k = (A_temp1 + I / gamma) \ (A_temp2 + y_k / gamma);
    z_k(find(z_k < 0)) = 1e-30;
    TEMP_B = 2 * z_k - gamma * lambda_2 * ( 1 ./ 2*(sqrt(z_k)) )  - y_k;
    x_k = (lambda_1 * (L' + L) + I / gamma) \ (TEMP_B / gamma);
    
    %改编
    if ~isempty(outIdx_ROI)
        y_k(outIdx_ROI) = 0;
        z_k(outIdx_ROI) = 0;
        x_k(outIdx_ROI) = 0;
    end
    %改编
    y_k = y_k - z_k + x_k;
    
    gamma = max(gamma/2, 0.9999*gamma);
    
    if iter_i == 1
        z_k0 = z_k;
    end
%     记录当前的预测的光源与真实光源分布之间的L2距离和余弦距离
    
    E_L2 = norm(A * z_k - surf_density)
    E_cos = sum(A * z_k .* surf_density) / (norm(A * z_k) * norm(surf_density))
    
    
    E_L2_sum = E_L2_sum + E_L2;
    E_cos_sum = E_cos_sum + E_cos;
    
    P_L2 = (1 ./ E_L2_sum) / sum(1 ./ E_L2_sum);
    P_cos = E_cos_sum / sum(E_cos_sum);
    P_err = (P_L2 + P_cos) / 2;
    
    P_X = z_k .* P_err / sum(z_k .* P_err);
    outIdx_ROI_length0 = length(outIdx_ROI);
    outIdx_ROI = find(P_X < inf_p*max(P_X));
    outIdx_ROI_length1 = length(outIdx_ROI);
    if outIdx_ROI_length1 - outIdx_ROI_length0 < 10
        inf_p = min(0.9, inf_p + 0.1);
    end
    iter_i = iter_i + 1;
    if E_cos < E_cos_0 || E_L2 > E_L2_0
        z_k = z_k0;
        break;
    end
    E_cos_0 = E_cos;
    E_L2_0 = E_L2;
    z_k0 = z_k;
end
end