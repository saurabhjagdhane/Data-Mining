clc;
close all;

x_train = importdata('Handwritten Digits/X_train.mat');
y_train = importdata('Handwritten Digits/y_train.mat');
x_test = importdata('Handwritten Digits/X_test.mat');
y_test = importdata('Handwritten Digits/y_test.mat');

%%% KNN with k=7 %%%
model=fitcknn(x_train,y_train,'NumNeighbors',7,'Distance','euclidean');
label_KNN=predict(model,x_test);
accuracy_KNN = classperf(y_test,label_KNN);
fprintf('KNN for Handwritten Digits, Accuracy= %.4f%%\n',accuracy_KNN.CorrectRate*100);

%%% SVM with a polynomial kernel of degree 2 %%%
temp = zeros(numel(y_train),1);
res = zeros(size(x_test,1),numel(unique(y_train)));
for i=1:numel(unique(y_train))
    for j = 1:numel(y_train)
        if(y_train(j) == i)
            temp(j) = 1;
        else
            temp(j) = -1;
        end
    end
    model = fitcsvm(x_train, temp, 'KernelFunction', 'polynomial', 'Polynomialorder' , 2);
    label = predict(model, x_test);
    res(1:size(x_test,1),i) = label;
end  

label = zeros(1,size(x_test,1));
for i=1:size(x_test,1)
    for j=1:numel(unique(y_train))
        if(res(i,j)==1)
            label(i)=j;
        end
    end
end

accuracy_SVM=classperf(y_test,label);
fprintf('SVM for Handwritten Digits, Accuracy= %.4f%%\n',accuracy_SVM.CorrectRate*100);

%%% Feedforward neural net with 25 neurons %%%
target  = full(ind2vec(y_train')) ;
net = patternnet(25);
net = train(net,x_train',target);
y = net(x_test');
classes = vec2ind(y);
accuracy_ANN=classperf(y_test,classes);
fprintf('ANN for Handwritten Digits, Accuracy= %.4f%%\n',accuracy_ANN.CorrectRate*100);

%%% Ensemble %%%
ensemble = [label_KNN';label;classes];
label_ensemble = zeros(size(x_test,1),1);
for i=1:size(x_test,1)
    label_ensemble(i) = mode(ensemble(:,i));
end 
accuracy_ensemble=classperf(y_test,label_ensemble);
fprintf('Ensemble for Handwritten Digits, Accuracy= %.4f%%\n',accuracy_ensemble.CorrectRate*100);