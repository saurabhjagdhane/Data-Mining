clc;
close all;

seeds = importdata('Clustering/seeds.txt');

% For each value of k, run the algorithm with 10 random initializations of the centroids.
sse1=zeros(1,10);
sse2=zeros(1,10);
sse3=zeros(1,10);

for i=1:10
    [sse1(i)] = mykmeans(seeds,3);
    [sse2(i)] = mykmeans(seeds,5);
    [sse3(i)] = mykmeans(seeds,7);
end

    fprintf('\nRandom initialization of centroids: %d times\n',i);
    fprintf('Average SSE for k=3 is %.4f\n',mean(sse1));
    fprintf('Average SSE for k=5 is %.4f\n',mean(sse2));
    fprintf('Average SSE for k=7 is %.4f\n',mean(sse3));