clc;
close all;

X_train=importdata('VidTIMIT/X_train.mat');
y_train=importdata('VidTIMIT/y_train.mat');
X_test=importdata('VidTIMIT/X_test.mat');

y_train = transpose(y_train);
temp = zeros(3500,1);
res = zeros(1000,25);
for i=1:25
    for j = 1:numel(y_train)
        if(y_train(j) == i)
            temp(j) = 1;
        else
            temp(j) = -1;
        end
    end
    model = fitcsvm(X_train, temp, 'KernelFunction', 'polynomial', 'Polynomialorder' , 2);
    label = predict(model, X_test);
    res(1:1000,i) = label;
end  

label = zeros(1,1000);
for i=1:1000
    for j=1:25
        if(res(i,j)==1)
            label(i)=j;
        end
    end
end

y_test=importdata('VidTIMIT/y_test.mat');

accuracy_SVM_2c=classperf(y_test,label);

fprintf('SVM for VidTIMIT, Accuracy= %.4f%%\n',accuracy_SVM_2c.CorrectRate*100);