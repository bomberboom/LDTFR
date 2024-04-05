clear all;
clc;
close all;
%------ functions path ------%
addpath('.\general_function\');
addpath('.\manifold');
addpath('./npy-matlab-master/')

%------ global parameters setting ------%
% the number of intra-class neighbors
para.kw = 15;
% the number of inter-class neighbors
para.kb = 10;
% tradoff parameter C
para.c = 0.2;
% the size of dimensionality-reduced CovD
para.newDim = 30;

%------ path setting ------%
path_data.train_data_path = './T&T_vgg16_train.npy';
path_data.train_label_path = './T&T_vgg16_train_label.npy';
path_data.test_data_path = './T&T_vgg16_test.npy';
path_data.test_label_path = './T&T_vgg16_test_label.npy';

path_work = '.\result\';

if ~isdir(path_work) 
    mkdir(path_work); 
end


%------ feature preparation ------%
disp('!!! Feature Preparation!!!!!');
covD_Struct=feature_OATH(path_data,para.newDim);

%------ classification ------%
disp('!!! Classification!!!!!');
result = SVM_classifier_traintest(covD_Struct,para);
save([path_work,'T&T_vgg16.mat'],'result');
disp('!!! Results has been recorded!!!!!');
return