function [covD_Struct,R,feature_21_train,feature_21_test]=feature_OATH(path_data,newDim)

train_data=readNPY(path_data.train_data_path);
test_data=readNPY(path_data.test_data_path);
train_label=readNPY(path_data.train_label_path);
test_label=readNPY(path_data.test_label_path);
[~, feature_map_size, ~ ,SPD_size] = size(train_data);

%% generat
for i=1:length(train_data)
    temp_data=train_data(i,:,:,:);
    I_new=reshape(temp_data,[feature_map_size*feature_map_size,SPD_size]);
    temp_SPD = cov(I_new);
    trn_X(:,:,i)=SPDproject(temp_SPD);
    temp_SPD=SPDproject(temp_SPD);
end
for i=1:length(test_data)
    temp_data=test_data(i,:,:,:);
    I_new=reshape(temp_data,[feature_map_size*feature_map_size,SPD_size]);
    temp_SPD = cov(I_new);
    tst_X(:,:,i)=SPDproject(temp_SPD);
end

trn_y=double(train_label)+1;
tst_y=double(test_label+1);
trn_X=double(trn_X);
tst_X=double(tst_X);

covD_Struct.trn_X=trn_X;
covD_Struct.trn_y=trn_y;
covD_Struct.tst_X=tst_X;
covD_Struct.tst_y=tst_y;

covD_Struct.n=SPD_size;

end
