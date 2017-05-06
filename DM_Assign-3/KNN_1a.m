clc;
close all;

X_train=importdata('Human Activity Recognition/X_train.txt');
y_train=importdata('Human Activity Recognition/y_train.txt');
X_test=importdata('Human Activity Recognition/X_test.txt');

%classify KNN%
%trainedModel=knnclassify(X_test,X_train,y_train,5,'euclidean','nearest');
model=fitcknn(X_train,y_train,'NumNeighbors',5,'Distance','euclidean');
label=predict(model,X_test);

y_test=importdata('Human Activity Recognition/y_test.txt');

accuracy_KNN_1a=classperf(y_test,label);

fprintf('KNN for Human Activity Recognition, Accuracy= %.4f%%\n',accuracy_KNN_1a.CorrectRate*100);