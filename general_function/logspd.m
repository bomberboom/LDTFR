function L=logpsd(X)
%可能是对应论文中exp和ln投影那部分，对应log(A),等式(11)，接下来也不用算log（A）了
   [u,d]=schur(X);%用了schur然后又只取特征值？和直接特征值分解是一个效果？
   d=diag(d);
   d(d<=0)=1;
   L=u*diag(log(d))*u';
end
