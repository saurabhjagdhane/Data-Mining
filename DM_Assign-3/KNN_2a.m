clc;
close all;

X_train=importdata('VidTIMIT/X_train.mat');
y_train=importdata('VidTIMIT/y_train.mat');
X_test=importdata('VidTIMIT/X_test.mat');

%classify KNN%
%trainedModel=knnclassify(X_test,X_train,y_train,5,'euclidean','nearest');
model=fitcknn(X_train,y_train,'NumNeighbors',5,'Distance','euclidean');
label=predict(model,X_test);

y_test=importdata('VidTIMIT/y_test.mat');

accuracy_KNN_2a=classperf(y_test,label);

fprintf('KNN for VidTIMIT, Accuracy= %.4f%%\n',accuracy_KNN_2a.CorrectRate*100);