%algo is clustering algo for each value of k
function[sse_new]=mykmeans(seeds,k)
N=size(seeds,1);
iterations = 0;
sse_before = 0;
sse_new=0;
centroids = seeds(randsample(N,k),:);

while 1
    iterations = iterations +1;
    dist=pdist2(centroids, seeds, 'euclidean');
    [distanceToNearestCluster, ClusterID]=min(dist,[],1);
    sse_new=0;
    
    for i=1:k
        indx=ClusterID==i;
        distSpecificToCluster = distanceToNearestCluster(indx);
        sse_new=sse_new+sumsqr(distSpecificToCluster);
    end
    
    if iterations>100 || abs(sse_before-sse_new)<0.001
        break; %break condition for while(1)
    end
    
    %before moving to next iteration store sse
    sse_before = sse_new;
    
    for i=1:k
       %calculate average here
       indx=ClusterID==i;
       centroids(i,:)=mean(seeds(indx,:),1);
       %and assign new centroid 
    end
end
