% achieve dimansionality reduction, train SVM classifier and inference
% input: 
%   covD_Struct:SPD and corresponding label
%   para:hyperparameters
% output:
%   result:classification results
% SPD dimension reduction implementation by Mehrtash Harandi (mehrtash.harandi at gmail dot com)
% adapted by Hu et al., 2024

function result = DR_SVM_classifier(covD_Struct,para)
    train_label = covD_Struct.trn_y;
    test_label_true = covD_Struct.tst_y;
    class_num = max(covD_Struct.trn_y);

    %% dimensionality reduction
    [TL_trnX,TL_tstX,U]=DR(covD_Struct,para);
    for countvariable=1:1:length(TL_trnX)
        log_TL_trnX(:,:,countvariable)=logspd(TL_trnX(:,:,countvariable));
    end
    for countvariable=1:1:length(TL_tstX)
        log_TL_tstX(:,:,countvariable)=logspd(TL_tstX(:,:,countvariable));
    end
    [train_data,test_data]=spd2vector(log_TL_trnX,log_TL_tstX);

    
    %% normalization
    ww = size(train_data,1);
    AA = max(train_data);
    BB = min(train_data);
    max_f = repmat(AA,[ww,1]);
    min_f = repmat(BB,[ww,1]);
    train_data = (train_data-min_f)./(max_f-min_f);   
    ee = size(test_data,1);
    max_f = repmat(AA,[ee,1]);
    min_f = repmat(BB,[ee,1]);
    test_data = (test_data-min_f)./(max_f-min_f);
    
    test_data(test_data<0)=0;
    test_data(test_data>1)=1;
    %% SVM
    [model class_num]= SupportVectorTrain(train_data,train_label,4,class_num);
    test_label = SupportVector(test_data,model,class_num);
    [max_val max_pos] = max(test_label');
    test_label_SVM = max_pos';
    %% show results
    clear train
    for j = 1:length(train_data)
        train(j).feature = train_data(j,:);
        train(j).kind = train_label(j);
    end
    result.train = train;
    
    clear test
    for j =1:size(test_data,1)
        test(j).feature = test_data(j,:);
        test(j).kind = test_label_true(j);
        test(j).SVM = test_label_SVM(j);
    end
    result.test = test;
    
    %%confused images
    clear diff
    ss = find(test_label_true~=test_label_SVM);
    if length(ss)~=0
        for k =1:length(ss)
            diff(k).feature = test(ss(k)).feature;
            diff(k).kind = test(ss(k)).kind;
            diff(k).SVM = test(ss(k)).SVM;
        end
        result.diff = diff;
    else
        result.diff = [];
    end
    
    result.U=U;
    %confusioin matrix
    for j = 1:class_num
        for g = 1:class_num
            confusion_num(j,g) = numel(find(test_label_true==j&test_label_SVM==g));
        end
    end
    acc= 1-sum(double(logical(test_label_true - test_label_SVM~=0)))/length(test_label_true);
    HH=repmat(sum(confusion_num(:,:),2),[1,class_num]);
    confusion_rat(:,:) = confusion_num(:,:)./HH;
    result.acc = acc;
    result.confusion_num = confusion_num(:,:);
    result.confusion_rat = confusion_rat(:,:);
    result.test_label=test_label(:,:);
    fprintf(['The accuracy is: %.2f%%\r\n'],acc*100);
end




