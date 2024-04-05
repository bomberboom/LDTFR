function [feature_train,feature_test]=spd2vector(TL_trnX,TL_tstX)
%addpath('C:\本科毕业设计\bishe\general_function\')
parfor i=1:length(TL_trnX)
    feature_train(i,:)=map2IDS_vectorize(TL_trnX(:,:,i),0);
end
parfor i=1:length(TL_tstX)
    feature_test(i,:)=map2IDS_vectorize(TL_tstX(:,:,i),0);
end
end