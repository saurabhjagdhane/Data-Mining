clc;
close all;

x_train = importdata('Multi Label Scene Data/X_train.mat');
y_train = importdata('Multi Label Scene Data/y_train.mat');
x_test = importdata('Multi Label Scene Data/X_test.mat');
y_test = importdata('Multi Label Scene Data/y_test.mat');

%%% SVM with polynomial kernel with parameter 2 %%%
labels=zeros(size(x_test,1),size(y_train,2));
for i=1:size(y_train,2)
    model = fitcsvm(x_train, y_train(:,i), 'KernelFunction', 'polynomial', 'Polynomialorder' , 2);
    labels(:,i) = predict(model, x_test);
end

numerator = bsxfun(@and, labels, y_test);
denominator = bsxfun(@or, labels, y_test);
% column vector containing sum of each row %
numerator_sum = sum(numerator,2); 
denominator_sum = sum(denominator,2);
accuracy = bsxfun(@rdivide, numerator_sum, denominator_sum);
fprintf('SVM with polynomial kernel with parameter 2, Accuracy= %.4f%%\n',mean(accuracy)*100);

%%% SVM with Gaussian kernel with parameter 2 %%%
labels=zeros(size(x_test,1),size(y_train,2));
for i=1:size(y_train,2)
    model = fitcsvm(x_train, y_train(:,i), 'KernelFunction', 'Gaussian', 'KernelScale', 'auto');
    labels(:,i) = predict(model, x_test);
end

numerator = bsxfun(@and, labels, y_test);
denominator = bsxfun(@or, labels, y_test);
% column vector containing sum of each row %
numerator_sum = sum(numerator,2); 
denominator_sum = sum(denominator,2);
accuracy = bsxfun(@rdivide, numerator_sum, denominator_sum);
fprintf('SVM with Gaussian kernel with parameter 2, Accuracy= %.4f%%\n',mean(accuracy)*100);