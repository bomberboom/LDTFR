% º∆À„Grassmann Manifold…œµƒæ‡¿Î
% PF£∫ Project F-normæ‡¿Î
% FS£∫ Fubini-Studyæ‡¿Î
% BC£∫ Binet-Cauchyæ‡¿Î
% PK£∫ Project kernelæ‡¿Î
% BCK£∫Binet-Cauchy∫Àæ‡¿Î
% 2020-8-1, NUDT in Nanjing, ver 0.0.1

% gm_p1,gm_p2: two points on Grassmann Manifold

function distance = grassmann_distance(gm_p1, gm_p2, metric_flag)

    [~, n] = size(gm_p1);
    
    d = 0;
    switch (metric_flag)
        case 1 
            %'PF'
            d = 1.0/sqrt(2) * norm(gm_p1 * gm_p1'- gm_p2 * gm_p2', 'fro');
            d = d * d;
        case 2
            %'FS'
            d = acos(abs(det(gm_p1' * gm_p2)));
            d = d * d;
        case 3 
            %'BC'
            d = 2 - 2 * abs(det(gm_p1' * gm_p2));
          
        case 4
            %'PK'
            d = 2 * n - 2 * norm(gm_p1' * gm_p2, 'fro')^2 ;
            
        case 5
            %'BCK'
            d = det(gm_p1' * gm_p2 * gm_p2' * gm_p1);
            
        otherwise
            error('The metric is not implemented.');
    end %end switch

    distance = d;
    
return