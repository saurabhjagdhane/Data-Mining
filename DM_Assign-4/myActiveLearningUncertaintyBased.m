function [accuracy_vector] = myActiveLearningUncertaintyBased(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix, unlabeledLabels)
num_classes=numel(unique(trainingLabels));
N=50; % Number of iterations
k=10; % batch size

accuracy_vector=zeros(1,N);

%%%%% ..........Looping starts here........%%%%
for loop=1:N
    
    %%% Step 1: Train a machine learning model using the current training set%%%
    trained_weights1 = train_LR_Classifier(trainingMatrix,trainingLabels,num_classes);

    %%% Step 2: Apply the model on the test set and obtain the accuracy%%%
    probabilityVector1=zeros(size(testingMatrix,1),num_classes);
    probabilityVector1_labels=zeros(size(testingMatrix,1),1);
    for i=1:size(testingMatrix,1)
        probabilityVector1(i,:) = test_LR_Classifier(testingMatrix(i,:),trained_weights1,num_classes);
        [~,probabilityVector1_labels(i)]=max(probabilityVector1(i,:));
    end

    accuracy = classperf(testingLabels,probabilityVector1_labels);
    accuracy_vector(1,loop)=accuracy.CorrectRate*100;
    % fprintf('Accuracy= %.4f\n',accuracy.CorrectRate*100);
    
    probabilityVector1_new=zeros(size(unlabeledMatrix,1),num_classes);
    entropyMatrix=zeros(size(unlabeledMatrix,1),1);
    % probabilityVector1_labels_new=zeros(size(randomSelection,1),1);
    for i=1:size(unlabeledMatrix,1)
        probabilityVector1_new(i,:) = test_LR_Classifier(unlabeledMatrix(i,:),trained_weights1,num_classes);
        % [~,probabilityVector1_labels_new(i)]=max(probabilityVector1_new(i,:));
        entropyMatrix(i)=-sum(probabilityVector1_new(i,:).*log2(probabilityVector1_new(i,:)));
    end
    
    [~,sortingIndices] = sort(entropyMatrix,'descend');
    maxValueIndices = sortingIndices(1:k);

    % Step 4 (Human expert labeling)
    trainingLabels=[trainingLabels;unlabeledLabels(maxValueIndices)];
    unlabeledLabels=removerows(unlabeledLabels,'ind',maxValueIndices);
    
    % Step 5: Remove samples and add them to current training set
    trainingMatrix=[trainingMatrix;unlabeledMatrix(maxValueIndices,:)];
    unlabeledMatrix=removerows(unlabeledMatrix,'ind',maxValueIndices);
    %fprintf('%d\n',size(unlabeledLabels,1));
end
end

