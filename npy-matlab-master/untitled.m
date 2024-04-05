clear
close all
clc
addpath('.\npy-matlab\')


data_feature=readNPY('D:\research\Inception4\vgg-tensorflow-master\extracted_features_mixed1_OATH_inceptionv3.npy');
GR=moduleresult2feature_gr(data_feature);
[W,H,D,C]=size(data_feature);



