function tmpDist = pair_Grassman_distance(Set1,Set2,Metric_Flag) %#ok<*FNDEF>



l1 = size(Set1,3);
l2 = size(Set2,3);
tmpDist = zeros(l2,l1);%%原(l1,l2)



for tmpC1 = 1:l1
        X = Set1(:,:,tmpC1);
        for tmpC2 = 1:l2
            Y = Set2(:,:,tmpC2);
            tmpDist(tmpC2,tmpC1) = grassmann_distance(X, Y, Metric_Flag);
            
            if  (tmpDist(tmpC2,tmpC1) < 1e-10)
                tmpDist(tmpC2,tmpC1) = 0.0;
            end
           
        end
end
    
return