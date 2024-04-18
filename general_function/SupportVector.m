function test_label = SupportVector(test_data,model,class_num)
%%%%   Input:       %%%%%%%%%%%%%%%%
%%%%    test_data：输入样本特征 %%%%
%%%%    class_num：总类别数     %%%%
%%%%    model：训练好的svm模型  %%%%

    [num_test fea_num] = size(test_data);
    test_label = zeros(num_test,class_num);
    for k = 1:num_test
        num = zeros(1,class_num);
        for i = 1:class_num
            for j = 1:i-1
                [G,score] = predict(model{i,j},test_data(k,:));%第i类和第j类更可能是哪一类。G=1，为i类；G=-1，为j类
                if G == 1
                    num(i) = num(i)+1;
                end
                if G == -1
                    num(j) = num(j)+1;
                end
            end
        end
        test_label(k,:) = num;
    end

return