# LDTFR
Classification of Ground-based Auroral Images by Learning Deep Tensor Feature Representation on Remannian Manifold
## Getting Start
### Datasets 
Dataset 1 used in our paper is available at https://dataverse.no/dataset.xhtml?persistentId =doi:10.18710/SSA38J. Dataset 2 can be found at http://tid.uio.no /plasma/oath/. 
### Dependencies
* Keras==2.3.1
* Tensorflow==2.0.0
* Scipy
* Os
* numpy
### Running
Notes: This project is excuted on python and matlab, first extract feature on python by feature_extraction.py, then optimize _W_ and train a multi-class SVM classifier and inference by mydemo_Train_test.m  
  
#### feature extraction：
Run feature_extraction.py to generate train_data.npy, test_data.npy, train_label.npy, and test_label.npy  
We also provide the extracted feature of dataset 1 in https://pan.baidu.com/s/1cW5cCUTzu4K04z5nIvnePw?pwd=0306 


#### dimensionality reduction and classification：
Run mydemo_TrainTest.m

