% Author:
% - Zeming, Zhou
%
% 2020-8-1, ver 0.0.1
% ref: Joint Normalization and Dimensionality Reduction on Grassmannian: A
% Generalized Persepctive
% Tianci Liu, Zelin Shi, and Yunpeng Liu. IEEE Signal Processing Letters,
% Vol.25, No. 6, 2018

% parameters:
% U: D*d; U':d*D
% gnD_Struct:
% .X: D*n, 
% .y: label
% .r: low dimension
% .Metric_Flag: metric flag of grassmannian manifold
% .G(i,j):affinity matrix element at (i,j)
% outCost: cost function
% outGrad: Riemannian gradient

function [outCost,outGrad,gnD_Struct] = supervised_GR_CostGrad(U,gnD_Struct)

% cost function
outCost = 0;
% Riemann gradient
dF = zeros(size(U));

% sample number
nPoints = length(gnD_Struct.y);

% n: number of orth vector of d-dimension
[rows, cols] = size(gnD_Struct.X(:,:,1));

% compute QR decomposition of U'* gnD_Struct.X
% UXR: r*n*samples
% UXR(:,:,i)->Qi; UXR(:,:,j)->Qj
XR = zeros(rows,cols,nPoints);
Q = zeros(gnD_Struct.r,cols,nPoints);
S = zeros(size(U,2),size(U,2),nPoints);

metric_flag = gnD_Struct.Metric_Flag;

for tmpC1 = 1:nPoints
     [tmpQ,R] = qr(U'*gnD_Struct.X(:,:,tmpC1), 0);
     XR(:,:,tmpC1) = gnD_Struct.X(:,:,tmpC1) / R;  %Q=W'XR-^-1%%%%%%%%%%
     Q(:,:,tmpC1) = tmpQ;
     S(:,:,tmpC1) = (eye(size(U,2)) - tmpQ * tmpQ')';
end


eGrad_ij = 0;
% compute the cost function and the E-gradient

for i = 1:nPoints
    
    for j = 1:nPoints
        
        if (gnD_Struct.G(i,j) == 0)
            continue;
        end
        
        % X1, X2: low dimensional point on Grassmannian manifold. X*R^-1 ==> X and U == W in the paper 
        X1 = XR(:,:,i);
        X2 = XR(:,:,j);
        Q1 = Q(:,:,i);
        Q2 = Q(:,:,j);
        dQ1 = 2 * (Q1 - Q2 * Q2' * Q1);
        dQ2 = 2 * (Q2 - Q1 * Q1' * Q2);
        S1 = S(:,:,i);
        S2 = S(:,:,j);
        britl_Q1 = tril(Q1' * dQ1,-1) - tril((Q1' * dQ1)',-1);
        britl_Q2 = tril(Q2' * dQ2,-1) - tril((Q2' * dQ2)',-1);
        % 计算两点距离的平方
        outCost = outCost + gnD_Struct.G(i,j)*grassmann_distance(Q1, Q2, metric_flag);  
        
        
        
        switch (metric_flag)

            case 1
                %'PF'
                %T0 = U' * (X1 * X1' - X2 * X2') * U;
                %eGrad_ij = eGrad_ij + 4 * (X1 * X1' - X2 * X2') * U * T0;
                %eGrad_ij = 2 * X1 * Q1' * (Q1 * Q1' - Q2 * Q2')-2 * X2 * Q2' * (Q1 * Q1' - Q2 * Q2');
                %A = X1 * X1' - X2 * X2';
                %eGrad_ij = 2 * A * U * U' * A * U;
                
                eGrad_ij = X1 * ((S1 * dQ1) + Q1 * britl_Q1)' + X2 * ((S2 * dQ2) + Q2 * britl_Q2)';
                
            case 2
                %'FS'
                T0 = X2' * U * U' * X1;
                t1 = det(T0);
                t2 = abs(t1);
                t3 = -0.5;
                T4 = X1' * U * U' * X2;
                t5 = det(T4);
                t6 = abs(t5);
                
                adj_T0 = inv(T0) * det(T0);
                adj_T4 = inv(T4) * det(T4);
                
                eGrad_ij = eGrad_ij + 2 * (1 - t2 * t2)^t3 * acos(t2) * sign (t1) * X1 * adj_T0 * X2' * U;
                eGrad_ij = eGrad_ij + 2 * (1 - t6 * t6)^t3 * acos(t6) * sign (t5) * X2 * adj_T4 * X1' * U;
                eGrad_ij = -eGrad_ij;
                
            case 3
                %'BC'
                T0 = X2' * (U * U') * X1;
                T1 = X1' * (U * U') * X2;
                
                adj_T0 = inv(T0) * det(T0);
                adj_T1 = inv(T1) * det(T1);
                
                eGrad_ij = eGrad_ij + sign(det(T0)) * X1 * adj_T0 * X2';
                eGrad_ij = eGrad_ij + sign(det(T1)) * X2 * adj_T1 * X1';
                eGrad_ij = -2 * eGrad_ij;
                
            case 4
                %'PK'
               eGrad_ij = eGrad_ij + X1 * X1' * U * U' * X2 * X2' * U;
               eGrad_ij = eGrad_ij + X2 * X2' * U * U' * X1 * X1' * U;
               eGrad_ij = -4 * eGrad_ij;
               
            case 5
                %'BCK'
               T0 = X1' * U * U' * X2 * X2' * U * U' * X1;
               adj_T0 = inv(T0) * det(T0);
               eGrad_ij = eGrad_ij + X1 * adj_T0 * X1' * U * U' * X2 * X2' * U;
               eGrad_ij = eGrad_ij + X2 * X2' * U * U' * X1 * adj_T0 * X1' * U;
               eGrad_ij = 2 * eGrad_ij;
                
            otherwise
                error('The metric is not implemented.');
                
        end %end switch
        
        dF = dF + eGrad_ij * gnD_Struct.G(i,j);
        
    end % end j
end % end i


outGrad = (eye(size(U,1)) - U * U') * dF;


end

