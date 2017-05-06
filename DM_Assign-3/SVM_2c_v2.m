clc;
close all;

X_train=importdata('VidTIMIT/X_train.mat');
y_train=importdata('VidTIMIT/y_train.mat');
X_test=importdata('VidTIMIT/X_test.mat');

y_train = transpose(y_train);

temp = zeros(numel(y_train),1);
res = zeros(size(X_test,1),numel(unique(y_train)));
for i=1:numel(unique(y_train))
    for j = 1:numel(y_train)
        if(y_train(j) == i)
            temp(j) = 1;
        else
            temp(j) = -1;
        end
    end
    model = fitcsvm(X_train, temp, 'KernelFunction', 'polynomial', 'Polynomialorder' , 2);
    label = predict(model, X_test);
    res(1:size(X_test,1),i) = label;
end  

label = zeros(1,size(X_test,1));
for i=1:size(X_test,1)
    for j=1:numel(unique(y_train))
        if(res(i,j)==1)
            label(i)=j;
        end
    end
end

y_test=importdata('VidTIMIT/y_test.mat');

accuracy_SVM_2c=classperf(y_test,label);

fprintf('SVM for VidTIMIT, Accuracy= %.4f%%\n',accuracy_SVM_2c.CorrectRate*100);