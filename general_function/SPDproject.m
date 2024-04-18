function A = SPDproject(X)
%输入和输出不是完全一样的吗
Y = (X + X') / 2;
[V,D] = eig(Y);
% D(find(D(find(D ~= 0)< 1e-4))) = 1e-4;
d = diag(D);
d(find(d <1e-4)) = 0.1;%把太小的特征值设置为1e-4
% d(find(d <1e-4)) = 1;
D = diag(d);
A = V * D * V';

 %%加上平均值
%  A=A+mean(V,2)*(mean(V,2))';
%  A=[A,mean(V,2);mean(V,2)',1];

