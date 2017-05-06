clc;
close all;

X_train=importdata('Human Activity Recognition/X_train.txt');
y_train=importdata('Human Activity Recognition/y_train.txt');
X_test=importdata('Human Activity Recognition/X_test.txt');

temp = zeros(7352,1);
res = zeros(2947,6);
for i=1:6
    for j = 1:numel(y_train)
        if(y_train(j) == i)
            temp(j) = 1;
        else
            temp(j) = -1;
        end
    end
    model = fitcsvm(X_train, temp, 'KernelFunction', 'polynomial', 'Polynomialorder' , 2);
    label = predict(model, X_test);
    res(1:2947,i) = label;
end  

label = zeros(1,2947);
for i=1:2947
    for j=1:6
        if(res(i,j)==1)
            label(i)=j;
        end
    end
end

y_test=importdata('Human Activity Recognition/y_test.txt');

accuracy_SVM_1b=classperf(y_test,label);

fprintf('SVM for Human Activity Recognition, Accuracy= %.4f%%\n',accuracy_SVM_1b.CorrectRate*100);