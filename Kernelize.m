function [KTrain] = Kernelize(Train,anchor)
    [n,~]=size(Train);
    
   
    KTrain = sqdist(Train',anchor');
    sigma = mean(mean(KTrain,2));
    KTrain = exp(-KTrain/(2*sigma));  
    mvec = mean(KTrain);
    KTrain = KTrain-repmat(mvec,n,1);
    
end