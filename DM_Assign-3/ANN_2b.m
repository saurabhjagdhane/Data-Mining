clc;
close all;

X_train=importdata('VidTIMIT/X_train.mat');
y_train=importdata('VidTIMIT/y_train.mat');
X_test=importdata('VidTIMIT/X_test.mat');

target  = full(ind2vec(y_train)) ;
net = patternnet(25);
net = train(net,transpose(X_train),target);
y = net(X_test');
classes = vec2ind(y);

y_test=importdata('VidTIMIT/y_test.mat');

accuracy_ANN_2b=classperf(y_test,classes);

fprintf('ANN for VidTIMIT, Accuracy= %.4f%%\n',accuracy_ANN_2b.CorrectRate*100);