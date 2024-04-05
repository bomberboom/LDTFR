
for i =1:5824
    for j=1:10
        result(i,j)=sum(sample_new(:,j,i).*sample_new(:,j,i));
    end
end
