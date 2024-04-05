
function orth_X = orthGrassman_Point(U, X)

% sample number
[rows, cols, dims] = size(X);


% compute QR decomposition of U'* gnD_Struct.X
% UXR: r*n*samples
% UXR(:,:,i)->Qi; UXR(:,:,j)->Qj
[rows, ~] = size(U');
orth_X = zeros(rows,cols,dims);


for tmpC1 = 1:dims
     [~,R] = qr(U'* X(:,:,tmpC1), 0);
     orth_X(:,:,tmpC1) = U' * X(:,:,tmpC1) / R;  %Q=W'XR-^-1
end

return