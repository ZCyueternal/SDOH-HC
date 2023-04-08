function [MAP_result,training_time] = train_twostep(I_tr,L_tr,param,I_te,L_te,anchor,Binit,Vinit,Pinit,Sinit,h)

%% set the parameters
nbits = param.nbits; % length of the hash code
beta = param.beta;
alpha = param.alpha;
delta = param.delta;
chunk = param.chunk; % 2000
gama = param.gama;

Sinit = Sinit +diag(2+zeros(1,size(L_tr,2))); % diag are all 1;

sita = param.sita;
yita = param.yita;
epsilon = param.epsilon;

paramiter = param.paramiter; % 10
nq = param.nq;

% dataset = param.datasets;



%% get the dimensions of features
n = size(I_tr,1); % 16000   (because we delete the last 883 samples
dX = size(anchor,1);  % 1000 (1000*4096)
dY = size(L_tr,2);  % 24


%% initialization
% Calculate n1,n2
% nq = 200 or 400
% n1 = floor((nq/chunk)*nq);
n1 = param.n1;
n2 = nq-n1; 

MAP_result=zeros(1,floor(n/chunk));  

nmax = param.nmax; % For select points
A = zeros(n,dY); % 16000*

myindex = zeros(floor(n/chunk),nmax);  % 8*1000   8 rounds' first 1k points

SA = zeros(n,dY); % G 16000*
normytagA = ones(chunk,1);% 2000*1

normdiff = zeros(1,chunk); % 1*2000   2000 ge 2-norm value of (L-Y)

% copy this to the class-wise hash code (leibie haxi ma)
Y = h; % c*r

B = Binit;
V = Vinit;
P = Pinit;

S = Sinit; % This is c*c;

% a = (dY*(dY+2)+dY*sqrt(dY*(dY+2)))/4 + eps;
% a =1;
% fprintf('a=%f\n',a);

XKTest=Kernelize(I_te,anchor); % Da phi(XTest)

%% iterative optimization
for round = 1:floor(n/chunk) % 1:16000/2000=1:8
    fprintf('chunk %d: training starts. \n',round)
    e_id = (round-1)*chunk+1; % start of this chunk
    n_id = min(round*chunk,n); % end of this chunk
    if round == floor(n/chunk) 
        n_id = n;
    end
    
    % RBF kernel mapping
    X = Kernelize(I_tr(e_id:n_id,:),anchor);
    nsample = n_id-e_id+1;  % 2000   % Places 100000

    tic;
    
    fprintf('round = %d\n',round);
        
    if round == 1
%%        simple Tag and low-level feature
%         diff = L_tr(e_id:n_id,:);

%         diff = L_tr(e_id:n_id,:)-B(e_id:n_id,:)*Y';
        L_minus_BY = L_tr(e_id:n_id,:)-B(e_id:n_id,:)*Y';
        diff = L_minus_BY.^2;
        for j = 1:nsample %1:2000
            normdiff(j) = norm(diff(j,:),2); % 2-norm of (Y) 1*2000
        end
        [~,index] = sort(normdiff(:,1:nsample)); % get the index  (ascending,from low to high)
        % myindex floor(n/chunk)*1000
        myindex(round,:) = index(:,1:nmax); % first 1k index of the points add to this round

        % norm simple YTrain, norm XTrain(after kernel)
        for i =1:nsample %1:2000
            if norm(L_tr(i+(round-1)*chunk,:))~=0 % if current chunk's L's norm !=0
                normytagA(i,:)=norm(L_tr(i+(round-1)*chunk,:));% 2000-d column vector
            end
            if norm(X(i+(round-1)*chunk,:))~=0 % if current chunk's L's norm !=0
                normX(i,:)=norm(X(i+(round-1)*chunk,:));% 2000-d column vector (2000*1)
        
            end
        end
        
        % This is ||L||
        normytagA = repmat(normytagA,1,dY); % 2000*404
        normX = repmat(normX,1,dX); % 2000*1000
        
        % SA is G t arrow (Gt=Lt/||Lt||)
        SA(e_id:n_id,:) = L_tr(e_id:n_id,:)./normytagA; 
        % SX is X/|X|
        SX(e_id:n_id,:) = X(e_id:n_id,:)./normX;

        for iter = 1:paramiter

            % update V

            LTB = SA(e_id:n_id,:)'*B(e_id:n_id,:); % V15 also use this part, please save this sentence.
            XTB = SX(e_id:n_id,:)'*B(e_id:n_id,:);

        % V16 only use Sno and Sqn
        Qt =  beta*B(e_id:n_id,:);

%% eigenvalue decompositon(like DGH)         
%             Temp = Qt'*Qt-1/nsample*(Qt'*ones(nsample,1)*(ones(1,nsample)*Qt));
            Temp1 = Qt'*(eye(nsample)-1/nsample*ones(1,nsample)*ones(nsample,1))*Qt;
%             [~,Lmd,QQ] = svd(Temp);
            [~,Lmd,QQ] = svd(Temp1);
%             clear Temp
            clear Temp1
            idx = (diag(Lmd)>1e-6);
            Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
            % The Qt of Temp and PP is opposite. shi xiang fan de
            PP = (Qt-1/nsample*ones(nsample,1)*(ones(1,nsample)*Qt)) *  (Q / (sqrt(Lmd(idx,idx))));
            P_ = orth(randn(nsample,nbits-length(find(idx==1))));
            V(e_id:n_id,:) = sqrt(nsample)*[PP P_]*[Q Q_]';
            
            % update Y arrow
            G = sita*S'*Y + yita*L_tr(e_id:n_id,:)'*B(e_id:n_id,:)+epsilon*h; % G:c*r
            for k=1:3
                for place=1:nbits
                    bit=1:nbits;
                    bit(place)=[];
                    Y(:,place) = sign(nbits*G(:,place)   -   Y(:,bit)*B(e_id:n_id,bit)'*B(e_id:n_id,place)-yita*Y(:,bit)*Y(:,bit)'*Y(:,place));
                end  
            end
            
            % UPDATE B my method

            LTV = SA(e_id:n_id,:)'*V(e_id:n_id,:);
            XTV = SX(e_id:n_id,:)'*V(e_id:n_id,:);


            % V16 only use Sno and Sqn
            U = beta*V(e_id:n_id,:) + yita*nbits*L_tr(e_id:n_id,:)*Y;
            for time=1:3
                for location=1:nbits
                    bite=1:nbits;
                    bite(location)=[];
                    B(e_id:n_id,location) = (sign(U(:,location)-yita*B(e_id:n_id,bite)*Y(:,bite)'*Y(:,location)))';
                end  
            end

            
        end
        
        C1 = X'*X;  % C1=X'*X
        C2 = X'*B(e_id:n_id,:);  % C2=X'*B
        Old_B = B(e_id:n_id,:);
        Old_V = V(e_id:n_id,:);
        % update P
        P = pinv(C1+delta*eye(dX))*(C2);
        
        Qq = L_tr(myindex(1,1:nq),:);
        Xq = X(myindex(1,1:nq),:);
        
        Btemp = B(e_id:n_id,:);
        Bq = Btemp(myindex(1,1:nq),:);

    end
    
    if round >= 2
        
	    P_last = P;
        CC1 = C1;
        CC2 = C2;
        OOld_B = Old_B;
        OOld_V = Old_V;
        LLTB = LTB;
        XXTB = XTB;

        LLTV = LTV;
        XXTV = XTV;

        L_minus_BY = L_tr(e_id:n_id,:)-B(e_id:n_id,:)*Y';
        diff = L_minus_BY.^2;
        
        normdiff = zeros(1,chunk);
        for j = 1:nsample
            normdiff(j) = norm(diff(j,:),2);
        end
        [~,index] = sort(normdiff(:,1:nsample));
        myindex(round,:) = index(:,1:nmax);
        
        normytagA = ones(chunk,1);


        for i =1:nsample
            if norm(L_tr(i+(round-1)*chunk,:))~=0 % if current chunk's L's norm !=0
                normytagA(i,:)=norm(L_tr(i+(round-1)*chunk,:));% 2000-d column vector
            end
            if norm(X(i,:))~=0 % if current chunk's L's norm !=0
                normX(i,:)=norm(X(i,:));% 2000-d column vector (2000*1)
        
            end
        end
        normytagA = repmat(normytagA,1,dY); % 2000*
        
        SA(e_id:n_id,:) = L_tr(e_id:n_id,:)./normytagA; % Gt=Lt/||Lt||
        % SX is X/|X|
        SX = X./normX;
        
        for iter = 1:paramiter

             % update V
            % Notation C is Qt in paper
            
            LLqT = SA(e_id:n_id,:)*Qq';
            iii = find(LLqT==0);
            
            S_mnT = SX*Xq';
            S_mnT(iii) = -1;
            
           
            % V16, only use Sno, Sqn,
            Qt = beta*B(e_id:n_id,:)+gama*nbits*S_mnT*Bq;

             Temp = Qt'*Qt-1/nsample*(Qt'*ones(nsample,1)*(ones(1,nsample)*Qt));
             [~,Lmd,QQ] = svd(Temp); clear Temp  % Lmd is \sigma^2, which is 
             idx = (diag(Lmd)>1e-6);
             Q = QQ(:,idx); Q_ = orth(QQ(:,~idx)); % value of non-zero, value of zerp
             PP = (Qt-1/nsample*ones(nsample,1)*(ones(1,nsample)*Qt)) *  (Q / (sqrt(Lmd(idx,idx))));
             P_ = orth(randn(nsample,nbits-length(find(idx==1))));
             V(e_id:n_id,:) = sqrt(nsample)*[PP P_]*[Q Q_]';

            % update Y arrow
            G = sita*S'*Y + yita*L_tr(e_id:n_id,:)'*B(e_id:n_id,:)+epsilon*h; % G:c*r
            for k=1:3
                for place=1:nbits
                    bit=1:nbits;
                    bit(place)=[];
                    Y(:,place) = sign(nbits*G(:,place)    -     Y(:,bit)*B(e_id:n_id,bit)'*B(e_id:n_id,place)-yita*Y(:,bit)*Y(:,bit)'*Y(:,place));
                end  
            end
            
            % UPDATE B my method

        % V16 
%         U = beta*V(e_id:n_id,:) + yita*nbits*A(e_id:n_id,:)*Y + alpha*nbits*SA(e_id:n_id,:)*LLTV;
        U = beta*V(e_id:n_id,:) + yita*nbits*A(e_id:n_id,:)*Y + alpha*nbits*(SA(e_id:n_id,:)*LLTV-ones(1,nsample)'*(ones(1,e_id-1)*Old_V));
            
            for time=1:3
                for location=1:nbits
                    bite=1:nbits;
                    bite(location)=[];
                    B(e_id:n_id,location) = (sign(U(:,location)-yita*B(e_id:n_id,bite)*Y(:,bite)'*Y(:,location)))';
                end  
            end

        end
        
        LTB = LLTB+SA(e_id:n_id,:)'*B(e_id:n_id,:);
        XTB = XXTB+SX'*B(e_id:n_id,:);
        

        LTV = LLTV+SA(e_id:n_id,:)'*V(e_id:n_id,:);
        XTV = XXTV+SX'*V(e_id:n_id,:);
        
        Old_B = [OOld_B ; B(e_id:n_id,:)];
        Old_V = [OOld_V ; V(e_id:n_id,:)];

        C1_new = X'*X;
        C2_new = X'*B(e_id:n_id,:);
        
        % update P
        C1 = CC1+C1_new;
        C2 = CC2+C2_new;
        
        P = pinv(C1+delta*eye(dX))*(C2);   
	    
        % update Qq
        yindex = myindex(round,1:n2) + (round-1)*chunk;
        neighbor = L_tr(yindex,:);
        olddata = Qq(randsample(nq,n1),:);
        Qq = [olddata;neighbor];
        
        
        % update Xq
        xindex = myindex(round,1:n2);
        neighbor1 = X(xindex,:);
        olddata1 = Xq(randsample(nq,n1),:);
        Xq = [olddata1;neighbor1];
        
        % update Bq
        oldBq = Bq(randsample(nq,n1),:);  
        Btemp = B(e_id:n_id,:);
        Bq = [oldBq;Btemp(myindex(round,1:n2),:)]; 
       
    end
    training_time(1,round) = toc;
    
    fprintf('       : training ends, training time is %f,\nevaluation begins. \n',training_time(1,round));
    
    BxTest = compactbit(XKTest*P >= 0);
    BxTrain = compactbit(B(1:n_id,:) >= 0);
    DHamm = hammingDist(BxTest, BxTrain); % ntest * ntrain
    [~, orderH] = sort(DHamm, 2); % each row, from low to high
    
    % my mAP
    MAP  = mAP(orderH', L_tr(1:n_id,:), L_te);
    fprintf('       : evaluation ends, MAP is %f\n',MAP);
    MAP_result(1,round)=MAP;
    
    % another mAP calculation method
%     param.unsupervised = 0;
%     Aff = affinity([], [], L_tr(1:n_id,:), L_te, param);
%     param.metric = 'mAP';
%     res = evaluate(B(1:n_id,:) >= 0, XKTest*P >= 0, param, Aff);
%     MAP_result(1,round) = res;


end
