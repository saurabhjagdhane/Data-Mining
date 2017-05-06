function [accuracy_vector] = myActiveLearningRandom(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix, unlabeledLabels)
num_classes=numel(unique(trainingLabels));
N=50; % Number of iterations
k=10; % batch size

accuracy_vector=zeros(1,N);

%%%%% ..........Looping starts here........%%%%
for loop=1:N
    
    %%% Step 1: Train a machine learning model using the current training set %%%
    trained_weights1 = train_LR_Classifier(trainingMatrix,trainingLabels,num_classes);

    %%% Step 2: Apply the model on the test set and obtain the accuracy %%%
    probabilityVector1=zeros(size(testingMatrix,1),num_classes);
    probabilityVector1_labels=zeros(size(testingMatrix,1),1);
    for i=1:size(testingMatrix,1)
        probabilityVector1(i,:) = test_LR_Classifier(testingMatrix(i,:),trained_weights1,num_classes);
        [~,probabilityVector1_labels(i)]=max(probabilityVector1(i,:));
    end

    accuracy = classperf(testingLabels,probabilityVector1_labels);
    accuracy_vector(1,loop)=accuracy.CorrectRate*100;
%   fprintf('Accuracy= %.4f\n',accuracy_vector(1,loop));
    
    % Step 3: Apply the model on the unlabeled set and select a batch of k unlabeled samples 

    % Random selection strategy
    [randomSelection,randomSelectionID] = datasample(unlabeledMatrix,k,'Replace',false);
    
    % Step 4 (Human expert labeling)
    trainingLabels=[trainingLabels;unlabeledLabels(randomSelectionID)];
    unlabeledLabels=removerows(unlabeledLabels,'ind',randomSelectionID);
    
    % Step 5: Remove samples and add them to current training set
    trainingMatrix=[trainingMatrix;randomSelection];
    unlabeledMatrix=removerows(unlabeledMatrix,'ind',randomSelectionID);
%   fprintf('%d\n',size(unlabeledLabels,1));
end
end

