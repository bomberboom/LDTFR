function result = SVM_classifier(data_feature,test_type,n_fold,numTrain,numRound)

if test_type %n-fold cross validation
    result = cross_valid(data_feature,n_fold,numRound);
else
    result = split_valid(data_feature,numTrain,numRound);
end
end

%------------------cross validation function------------%
function result = cross_valid(data_feature,n_fold,numRound)

len_feature_21=size(data_feature_Testing(1).feature_21,2);
samp_num = length(data_feature);
feature_4 = zeros(samp_num,4);
feature_21 = zeros(samp_num,len_feature_21);
kind = zeros(samp_num,1);

for i = 1:samp_num
    feature_4(i,:) = data_feature(i).feature_GLCM;%texture feature
    feature_21(i,:) = data_feature(i).feature_21;%manifold feature
    kind(i,:) = data_feature(i).kind+1;
end

feature = [feature_4,feature_21];%Combining manifold and texture features
class_num = length(unique(kind));

for t = 1:numRound
    indices = crossvalind('Kfold',samp_num,n_fold);%给每个数据编上[1，10]之间的整数，数字相同的为一组
    for i = 1:n_fold
        try%当测试集或者训练集的类别数《class_num时，混淆矩阵大小与之前不一，无法赋值，所以用try保证代码能够运行
            %%生成训练集和测试集
            test_ind = (indices == i);%测试集编号
            train_ind = ~test_ind;%训练集编号
            train_feature = feature(train_ind,:);
            train_kind = kind(train_ind,:);
            test_feature = feature(test_ind,:);
            test_kind_true = kind(test_ind,:);
            %% normalization
            ww = size(train_feature,1);
            AA = max(train_feature);
            BB = min(train_feature);
            max_f = repmat(AA,[ww,1]);
            min_f = repmat(BB,[ww,1]);
            train_feature = (train_feature-min_f)./(max_f-min_f);
            
            ee = size(test_feature,1);
            max_f = repmat(AA,[ee,1]);
            min_f = repmat(BB,[ee,1]);
            test_feature = (test_feature-min_f)./(max_f-min_f);
            
            test_feature(test_feature<0)=0;
            test_feature(test_feature>1)=1;
            %% SVM
            [model,class_num]= SupportVectorTrain(train_feature,train_kind,1);
            test_label = SupportVector(test_feature,model,class_num);
            [max_val,max_pos] = max(test_label');%创建一个矩阵 A 并计算每列中的最大元素，以及这些元素在 A 中显示的行索引。找出最有可能的类别
            pre_label_SVM = max_pos';
            %% computing accuracy and confusion matrix
            ex(i,:)= 1-sum(double(logical(test_kind_true - pre_label_SVM~=0)))/length(test_kind_true);%overall accuracy
            test_confusion_mat(:,:,i)=confusionmat(test_kind_true,pre_label_SVM);%%confusion matrix
            P = repmat(sum(test_confusion_mat(:,:,i),2),[1,class_num]);%sum(A,2) 是包含每一行总和的列向量。
            test_confusion_rat(:,:,i)=test_confusion_mat(:,:,i)./P;%%confusion matrix by accuracy？
            cross(i).ex = ex(i,:);
            cross(i).test_confusion_mat = test_confusion_mat(:,:,i);
            cross(i).test_confusion_rat = test_confusion_rat(:,:,i);
        end
            i=i-1;
    end
    mean_ex(t,:) = mean(ex);
    result(t).mean_ex = mean_ex(t,:);
    result(t).cross = cross;
    mean_test_confusion_mat(:,:,t) = mean(test_confusion_mat,3);%在第三个维度的均值，把十个test_confusion_mat相加/10
    result(t).mean_test_confusion_mat = mean_test_confusion_mat(:,:,t);
    mean_test_confusion_rat(:,:,t) = mean(test_confusion_rat,3);
    result(t).mean_test_confusion_rat = mean_test_confusion_rat(:,:,t);
    fprintf(['The accuracy of %g round is: %.2f%%\r\n'],t,mean_ex(t,:)*100);
end
result(1).total_ex = mean(mean_ex);
result(1).total_confusion_mat = mean(mean_test_confusion_mat,3);
result(1).total_confusion_rat = mean(mean_test_confusion_rat,3);
fprintf(['The overall accuracy of %.2f%%\r\n'],mean(mean_ex)*100);

end

function result = split_valid(data_feature,numTrain,numRound)

len_feature_21=size(data_feature(1).feature_21,2);
samp_num = length(data_feature);
feature_4 = zeros(samp_num,4);%texture-feature，GLCM是四维
feature_21 = zeros(samp_num,len_feature_21);%manifold-feature，21维
kind = zeros(samp_num,1);%各云图的类型，通过编号表示

for i = 1:samp_num
    feature_4(i,:) = data_feature(i).feature_GLCM;
    feature_21(i,:) = data_feature(i).feature_21;
    kind(i,:) = data_feature(i).kind+1;%避免第0类无法索引
end

feature = [feature_21];%融合texture和manifold feature
class_num = length(unique(kind)); %总的云图类别数量

ex = zeros(numRound,1);
confusion_rat = zeros(class_num,class_num,numRound);%？
confusion_num = zeros(class_num,class_num,numRound);% %
for i = 1:numRound
    [train_feature,train_kind,test_feature,test_kind,train_index(:,i),test_index(:,i)] = sample_extract_new(feature,kind,numTrain);
    result(i).train_index = train_index(:,i);
    result(i).test_index= test_index(:,i);
    %%  split training set and testing set
    train_data = train_feature;
    train_label = train_kind;
    test_data = test_feature;
    test_label_true = test_kind;
    
    %% normalization
    ww = size(train_data,1);
    AA = max(train_data);%包含每一列的最大值的行向量
    BB = min(train_data);%包含每一列的最小值的行向量
    max_f = repmat(AA,[ww,1]);%repmat:将所给向量（矩阵）当作一个整体，按照规定排列方法生成
    min_f = repmat(BB,[ww,1]);
    train_data = (train_data-min_f)./(max_f-min_f);   %归一化
    ee = size(test_data,1);
    max_f = repmat(AA,[ee,1]);
    min_f = repmat(BB,[ee,1]);
    test_data = (test_data-min_f)./(max_f-min_f);
    
    test_data(test_data<0)=0;%会有值《0或者》1吗
    test_data(test_data>1)=1;
    %% SVM
    [model class_num]= SupportVectorTrain(train_data,train_label,4,6);
    test_label = SupportVector(test_data,model,class_num);
    [max_val max_pos] = max(test_label');
    test_label_SVM = max_pos';
    %% show results
    clear train
    for j = 1:length(train_feature)
        train(j).name = data_feature(train_index(j,i)).name;
        train(j).feature = train_feature(j,:);
        train(j).kind = train_kind(j);
    end
    result(i).train = train;
    
    clear test
    for j =1:length(test_index(:,i))
        test(j).name = data_feature(test_index(j,i)).name;
        test(j).feature = test_feature(j,:);
        test(j).kind = test_kind(j);
        test(j).SVM = test_label_SVM(j);
    end
    result(i).test = test;
    
    %%confused images
    clear diff
    ss = find(test_label_true~=test_label_SVM);
    if length(ss)~=0
        for k =1:length(ss)
            diff(k).name = test(ss(k)).name;
            diff(k).feature = test(ss(k)).feature;
            diff(k).kind = test(ss(k)).kind;
            diff(k).SVM = test(ss(k)).SVM;
        end
        result(i).diff = diff;
    else
        result(i).diff = [];
    end
    
    %confusioin matrix
    for j = 1:class_num
        for g = 1:class_num
            confusion_num(j,g,i) = numel(find(test_label_true==j&test_label_SVM==g));
        end
    end
    ex(i,:)= 1-sum(double(logical(test_label_true - test_label_SVM~=0)))/length(test_label_true);%overall accuracy
    HH=repmat(sum(confusion_num(:,:,i),2),[1,class_num]);
    confusion_rat(:,:,i) = confusion_num(:,:,i)./HH;
    result(i).ex = ex(i,:);
    result(i).confusion_num = confusion_num(:,:,i);
    result(i).confusion_rat = confusion_rat(:,:,i);
    fprintf(['The accuracy of %g round is: %.2f%%\r\n'],i,ex(i,:)*100);
end

result(1).ex_mean = mean(ex);%%overall accuracy by numRound
result(1).confusion_num_mean = mean(confusion_num,3);
result(1).confusion_rat_mean = mean(confusion_rat,3);%%confusioin matrix by numRound
fprintf(['The overall accuracy of %.2f%%\r\n'],mean(ex)*100);

end



