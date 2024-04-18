function [TL_trnX,TL_tstX,U]=DR(covD_Struct,para)
% Author:
% - Mehrtash Harandi (mehrtash.harandi at gmail dot com)
%
% This file is provided without any warranty of
% fitness for any purpose. You can redistribute
% this file and/or modify it under the terms of
% the GNU General Public License (GPL) as published
% by the Free Software Foundation, either version 3
% of the License or (at your option) any later version.


addpath ('./manifold/local_manopt')

Metric_Flag = 2; %1:AIRM, 2:Stein

%initializing training structure
trnStruct.X = covD_Struct.trn_X;
trnStruct.y = covD_Struct.trn_y;
trnStruct.n = covD_Struct.n;
trnStruct.nClasses = max(covD_Struct.trn_y);
trnStruct.r = para.newDim;
trnStruct.Metric_Flag = Metric_Flag;
newDim = para.newDim;

%Generating graph
nPoints = length(trnStruct.y);
trnStruct.G = generate_Graphs(trnStruct.X,trnStruct.y,para,Metric_Flag);

%- different ways of initializing, the first 10 features are genuine so
%- the first initialization is the lucky guess, the second one is a random
%- attempt and the last one is the worst possible initialization.

% U = orth(rand(trnStruct.n,trnStruct.r));
U = eye(trnStruct.n,trnStruct.r);
% U = [zeros(trnStruct.n-trnStruct.r,trnStruct.r);eye(trnStruct.r)];

% Create the problem structure.
manifold = grassmannfactory(covD_Struct.n,newDim);
problem.M = manifold;

% conjugate gradient on Grassmann

problem.costgrad = @(U) supervised_WB_CostGrad(U,trnStruct,para.c);
U  = conjugategradient(problem,U,struct('maxiter',50));

TL_trnX = zeros(newDim,newDim,length(covD_Struct.trn_y));
parfor tmpC1 = 1:nPoints
    TL_trnX(:,:,tmpC1) = U'*covD_Struct.trn_X(:,:,tmpC1)*U;
end
TL_tstX = zeros(newDim,newDim,length(covD_Struct.tst_y));
parfor tmpC1 = 1:length(covD_Struct.tst_y)
    TL_tstX(:,:,tmpC1) = U'*covD_Struct.tst_X(:,:,tmpC1)*U;
end

end