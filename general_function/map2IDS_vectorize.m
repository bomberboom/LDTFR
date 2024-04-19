function y = map2IDS_vectorize(inMat, map2IDS)
%将正定对称后的矩阵向量化，并且利用对称矩阵的性质将非对角线元素乘2**0.5
if map2IDS == 1
    inMat = logm(inMat);    
end
offDiagonals = tril(inMat,-1) * sqrt(2);%tril:仅提取主对角线下方的元素,得到非对角线元素的2**0.5
diagonals = diag(diag(inMat));
vecInMat = diagonals + offDiagonals; 
vecInds = tril(ones(size(inMat)));
map2ITS = vecInMat(:);%“：”将矩阵按列重组为n*1的新矩阵，第一列排完排第二列
vecInds = vecInds(:);
y = map2ITS(vecInds == 1) ;