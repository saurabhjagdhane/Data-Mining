clc;
close all;

%%% iteration vector %%%
iteration=reshape((1:50),[1 50]);

%%% Dataset-1 (MMI) %%%
trainingMatrix_1 = importdata('Active Learning/MMI/trainingMatrix_1.mat');
trainingLabels_1 = importdata('Active Learning/MMI/trainingLabels_1.mat');
testingMatrix_1 = importdata('Active Learning/MMI/testingMatrix_1.mat');
testingLabels_1 = importdata('Active Learning/MMI/testingLabels_1.mat');
unlabeledMatrix_1 = importdata('Active Learning/MMI/unlabeledMatrix_1.mat');
unlabeledLabels_1 = importdata('Active Learning/MMI/unlabeledLabels_1.mat');

fprintf('Please be patient it takes a while....Processing Dataset-1 (MMI) [Random sampling]\n');
[accuracy_1a]=myActiveLearningRandom(trainingMatrix_1,trainingLabels_1,testingMatrix_1,testingLabels_1,unlabeledMatrix_1,unlabeledLabels_1);

fprintf('Please be patient it takes a while....Processing Dataset-1 (MMI) [Uncertainty-based Sampling]\n');
[accuracy_1b]=myActiveLearningUncertaintyBased(trainingMatrix_1,trainingLabels_1,testingMatrix_1,testingLabels_1,unlabeledMatrix_1,unlabeledLabels_1);


%%% Dataset-2 (MMI) %%%
trainingMatrix_2 = importdata('Active Learning/MMI/trainingMatrix_2.mat');
trainingLabels_2 = importdata('Active Learning/MMI/trainingLabels_2.mat');
testingMatrix_2 = importdata('Active Learning/MMI/testingMatrix_2.mat');
testingLabels_2 = importdata('Active Learning/MMI/testingLabels_2.mat');
unlabeledMatrix_2 = importdata('Active Learning/MMI/unlabeledMatrix_2.mat');
unlabeledLabels_2 = importdata('Active Learning/MMI/unlabeledLabels_2.mat');

fprintf('Please be patient it takes a while....Processing Dataset-2 (MMI) [Random sampling]\n');
[accuracy_2a]=myActiveLearningRandom(trainingMatrix_2,trainingLabels_2,testingMatrix_2,testingLabels_2,unlabeledMatrix_2,unlabeledLabels_2);

fprintf('Please be patient it takes a while....Processing Dataset-2 (MMI) [Uncertainty-based Sampling]\n');
[accuracy_2b]=myActiveLearningUncertaintyBased(trainingMatrix_2,trainingLabels_2,testingMatrix_2,testingLabels_2,unlabeledMatrix_2,unlabeledLabels_2);


%%% Dataset-3 (MMI) %%%
trainingMatrix_3 = importdata('Active Learning/MMI/trainingMatrix_3.mat');
trainingLabels_3 = importdata('Active Learning/MMI/trainingLabels_3.mat');
testingMatrix_3 = importdata('Active Learning/MMI/testingMatrix_3.mat');
testingLabels_3 = importdata('Active Learning/MMI/testingLabels_3.mat');
unlabeledMatrix_3 = importdata('Active Learning/MMI/unlabeledMatrix_3.mat');
unlabeledLabels_3 = importdata('Active Learning/MMI/unlabeledLabels_3.mat');

fprintf('Please be patient it takes a while....Processing Dataset-3 (MMI) [Random sampling]\n');
[accuracy_3a]=myActiveLearningRandom(trainingMatrix_3,trainingLabels_3,testingMatrix_3,testingLabels_3,unlabeledMatrix_3,unlabeledLabels_3);

fprintf('Please be patient it takes a while....Processing Dataset-3 (MMI) [Uncertainty-based Sampling]\n');
[accuracy_3b]=myActiveLearningUncertaintyBased(trainingMatrix_3,trainingLabels_3,testingMatrix_3,testingLabels_3,unlabeledMatrix_3,unlabeledLabels_3);

accuracy_MMI_Random = (accuracy_1a+accuracy_2a+accuracy_3a)/3;
accuracy_MMI_UncertaintyBased = (accuracy_1b+accuracy_2b+accuracy_3b)/3;

figure(1);
plot(iteration,accuracy_MMI_Random); hold on;
plot(iteration,accuracy_MMI_UncertaintyBased);
title('MMI');
legend('Random sampling','Uncertainty-based Sampling','Location','southeast');
xlabel('Number of iterations');
ylabel('Accuracy in %');



%%% Dataset-1 (MindReading) %%%
trainingMatrix_MindReading1 = importdata('Active Learning/MindReading/trainingMatrix_MindReading1.mat');
trainingLabels_MindReading_1 = importdata('Active Learning/MindReading/trainingLabels_MindReading_1.mat');
testingMatrix_MindReading1 = importdata('Active Learning/MindReading/testingMatrix_MindReading1.mat');
testingLabels_MindReading1 = importdata('Active Learning/MindReading/testingLabels_MindReading1.mat');
unlabeledMatrix_MindReading1 = importdata('Active Learning/MindReading/unlabeledMatrix_MindReading1.mat');
unlabeledLabels_MindReading_1 = importdata('Active Learning/MindReading/unlabeledLabels_MindReading_1.mat');

fprintf('Please be patient it takes a while....Processing Dataset-1 (MindReading) [Random sampling]\n');
[accuracy_4a]=myActiveLearningRandom(trainingMatrix_MindReading1,trainingLabels_MindReading_1,testingMatrix_MindReading1,testingLabels_MindReading1,unlabeledMatrix_MindReading1,unlabeledLabels_MindReading_1);

fprintf('Please be patient it takes a while....Processing Dataset-1 (MindReading) [Uncertainty-based Sampling]\n');
[accuracy_4b]=myActiveLearningUncertaintyBased(trainingMatrix_MindReading1,trainingLabels_MindReading_1,testingMatrix_MindReading1,testingLabels_MindReading1,unlabeledMatrix_MindReading1,unlabeledLabels_MindReading_1);


%%% Dataset-2 (MindReading) %%%
trainingMatrix_MindReading2 = importdata('Active Learning/MindReading/trainingMatrix_MindReading2.mat');
trainingLabels_MindReading_2 = importdata('Active Learning/MindReading/trainingLabels_MindReading_2.mat');
testingMatrix_MindReading2 = importdata('Active Learning/MindReading/testingMatrix_MindReading2.mat');
testingLabels_MindReading2 = importdata('Active Learning/MindReading/testingLabels_MindReading2.mat');
unlabeledMatrix_MindReading2 = importdata('Active Learning/MindReading/unlabeledMatrix_MindReading2.mat');
unlabeledLabels_MindReading_2 = importdata('Active Learning/MindReading/unlabeledLabels_MindReading_2.mat');

fprintf('Please be patient it takes a while....Processing Dataset-2 (MindReading) [Random sampling]\n');
[accuracy_5a]=myActiveLearningRandom(trainingMatrix_MindReading2,trainingLabels_MindReading_2,testingMatrix_MindReading2,testingLabels_MindReading2,unlabeledMatrix_MindReading2,unlabeledLabels_MindReading_2);

fprintf('Please be patient it takes a while....Processing Dataset-2 (MindReading) [Uncertainty-based Sampling]\n');
[accuracy_5b]=myActiveLearningUncertaintyBased(trainingMatrix_MindReading2,trainingLabels_MindReading_2,testingMatrix_MindReading2,testingLabels_MindReading2,unlabeledMatrix_MindReading2,unlabeledLabels_MindReading_2);


%%% Dataset-3 (MindReading) %%%
trainingMatrix_MindReading3 = importdata('Active Learning/MindReading/trainingMatrix_MindReading3.mat');
trainingLabels_MindReading_3 = importdata('Active Learning/MindReading/trainingLabels_MindReading_3.mat');
testingMatrix_MindReading3 = importdata('Active Learning/MindReading/testingMatrix_MindReading3.mat');
testingLabels_MindReading3 = importdata('Active Learning/MindReading/testingLabels_MindReading3.mat');
unlabeledMatrix_MindReading3 = importdata('Active Learning/MindReading/unlabeledMatrix_MindReading3.mat');
unlabeledLabels_MindReading_3 = importdata('Active Learning/MindReading/unlabeledLabels_MindReading_3.mat');

fprintf('Please be patient it takes a while....Processing Dataset-3 (MindReading) [Random sampling]\n');
[accuracy_6a]=myActiveLearningRandom(trainingMatrix_MindReading3,trainingLabels_MindReading_3,testingMatrix_MindReading3,testingLabels_MindReading3,unlabeledMatrix_MindReading3,unlabeledLabels_MindReading_3);

fprintf('Please be patient it takes a while....Processing Dataset-3 (MindReading) [Uncertainty-based Sampling]\n');
[accuracy_6b]=myActiveLearningUncertaintyBased(trainingMatrix_MindReading3,trainingLabels_MindReading_3,testingMatrix_MindReading3,testingLabels_MindReading3,unlabeledMatrix_MindReading3,unlabeledLabels_MindReading_3);

accuracy_MindReading_Random = (accuracy_4a+accuracy_5a+accuracy_6a)/3;
accuracy_MindReading_UncertaintyBased = (accuracy_4b+accuracy_5b+accuracy_6b)/3;

figure(2);
plot(iteration,accuracy_MindReading_Random); hold on;
plot(iteration,accuracy_MindReading_UncertaintyBased);
title('MindReading');
legend('Random sampling','Uncertainty-based Sampling','Location','southeast');
xlabel('Number of iterations');
ylabel('Accuracy in %');


