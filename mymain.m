clear;
addpath(genpath('./'));

nbits_set=[16 32 64 96 128];

%% load dataset
fprintf('loading dataset...\n')

set = 'CIFAR10';
% set = 'MIRFlickr';
% set = 'NUS-WIDE';
% set = 'Places';

if strcmp(set,'MIRFlickr')
    load('../Datasets/MIRFLICKR.mat');
    I_tr = I_tr(1:18015,:);
    L_tr = L_tr(1:18015,:);
elseif strcmp(set,'NUS-WIDE')
    load('../Datasets/NUSWIDE10.mat');
    I_tr = I_tr(1:40000,:);
    L_tr = L_tr(1:40000,:);
elseif strcmp(set,'CIFAR10')
    load('../Datasets/cifar10-zcyucut-follow-FOH.mat');
    L_tr = L_tr_onehot;   L_te = L_te_onehot;
elseif strcmp(set,'Places')
%     load('../Datasets/Places205_AlexNet_fc7_PCA128');
%     L_tr = L_tr_onehot;   L_te = L_te_onehot;
    opt.dirs.data = '../Datasets/';
    DS = Datasets.places(opt,0);
    
    trainCNN = DS.Xtrain;
    testCNN = DS.Xtest;
    trainLabel = DS.Ytrain;
    testLabel = DS.Ytest;
    
    test = testCNN ./ sqrt(sum(testCNN .* testCNN, 2));  
    testLabelvec = full(ind2vec(testLabel')); % c x n
    
    train = trainCNN ./ sqrt(sum(trainCNN .* trainCNN, 2));  
    trainLabelvec = full(ind2vec(trainLabel')); %c x n
    
    I_tr = train; % 2444772
    I_te = test; % 4100
%     L_tr = trainLabel;
%     L_te = testLabel;

    L_tr_onehot = trainLabelvec'; % 2444772 x 205
    L_te_onehot = testLabelvec'; % 4100 x 205
    
    L_tr = L_tr_onehot;
    L_te = L_te_onehot;
    
    clear trainCNN testCNN trainLabel testLabel train test testLabelvec trainLabelvec L_tr_onehot L_te_onehot opt;
end

anchor=I_tr(randsample(2000,1000),:); %% random select 1000 sample from XTrain (1000*4096)

%% initialization
fprintf('initializing...\n')
if strcmp(set,'MIRFlickr')
% MIR
    param.alpha = 1; param.gama = param.alpha;
    
    param.beta = 1;
    param.delta = 1;
    param.sita = 10;
    param.yita = 1;
    param.epsilon = 10;

elseif strcmp(set,'NUS-WIDE')
    % NUSWIDE
    param.alpha = 1;  param.gama = param.alpha;

    param.beta = 10;
    param.delta = 1;
    param.sita = 10; param.epsilon = param.sita;
    param.yita = 10; 

elseif strcmp(set,'CIFAR10')
    % CIFAR10
    param.alpha = 1;     param.gama = param.alpha;
    param.beta = 1;  
    param.delta = 1; 
    param.sita = 0.1; param.epsilon = param.sita;
    param.yita = 100; 

end

param.datasets = set;

param.paramiter = 10;
if strcmp(set,'MIRFlickr')
    param.nq = 200; 
    param.n1 = 100;
    param.chunk = 2000;
    param.nmax = 1000;
elseif strcmp(set,'NUS-WIDE')
    param.nq = 400;
    param.n1 = 100;
    param.chunk = 10000;
    param.nmax = 1000;
elseif strcmp(set,'CIFAR10')
    param.nq = 200;
    param.n1 = 100;
    param.chunk = 2000;
    param.nmax = 1000;
elseif strcmp(set,'Places')
    param.nq = 4000;
    param.n1 = 1000;
    param.chunk = 100000;
    param.nmax = 10000;
end

%% model training
for bit=1:length(nbits_set)
    nbits=nbits_set(bit);
    Binit = sign(randn(size(I_tr,1), nbits));
    Vinit = randn(size(I_tr,1), nbits);
    Pinit = randn(1000, nbits);
    Sinit = zeros(size(L_tr,2),size(L_tr,2))-1;
    param.nbits=nbits;
    
    % randomly generate Teacher codebook
    if strcmp(param.datasets,'MIRFlickr')
        h = hadamard(512); % 404tags/ 24label
        h = h(randperm(size(L_tr,2)),randperm(nbits)); % 404*nbits
    elseif strcmp(param.datasets,'NUS-WIDE')
        h = hadamard(8192); % 5000
        h = h(randperm(size(L_tr,2)),randperm(nbits)); % 5000*nbits
    elseif strcmp(param.datasets,'CIFAR10')
        h = hadamard(256); % 10
        h = h(randperm(size(L_tr,2)),randperm(nbits)); % 5000*nbits
    elseif strcmp(param.datasets,'Places')
        h = hadamard(256); % 128
        h = h(randperm(size(L_tr,2)),randperm(nbits)); % 5000*nbits
    end

[ MAP(bit,:),training_time(bit,:)] = train_twostep(I_tr,L_tr,param,I_te,L_te,anchor,Binit,Vinit,Pinit,Sinit,h);

end 
