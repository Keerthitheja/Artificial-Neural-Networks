clc
clear all
close all
%My student number: r0773368
d1=8;
d2=7;
d3=7;
d4=6;
d5=3;

%Part 1
load("Data_Problem1_regression.mat");
TNew = (d1*T1 + d2*T3 + d3*T3 + d4*T4 + d5*T5)/(d1+d2+d3+d4+d5);
Y = TNew';
X = [X1,X2]';

data_subset = randperm(size(X,2),3000);
train_idx = data_subset(1:1000);
val_idx = data_subset(1001:2000);
train_val_idx = data_subset(1:2000);
test_idx = data_subset(2001:3000);

% training set
train_X = X(:,train_idx);
train_Y = Y(train_idx);

% validation set
train_val_X = X(:,train_val_idx);
train_val_Y = Y(:,train_val_idx);

% train_validation set
val_X = X(:,val_idx);
val_Y = Y(:,val_idx);

% test set
test_X = X(:,test_idx);
test_Y = Y(:,test_idx);

[xq,yq] = meshgrid(0:0.01:1, 0:0.01:1);
zq_train = griddata(train_X(1,:),train_X(2,:),train_Y,xq,yq);
figure
mesh(xq,yq,zq_train);
hold on
plot3(train_X(1,:),train_X(2,:),train_Y,'.','MarkerSize',10,'linewidth',2);
title('Training Data');
xlabel('X1');
ylabel('X2');
zlabel('Target (Tnew)')
legend('Surface','Scattered Points','Location','Best')
set(gca,'FontSize',20);

[xq,yq] = meshgrid(0:0.01:1, 0:0.01:1);
zq_train_val = griddata(train_val_X(1,:),train_val_X(2,:),train_val_Y,xq,yq);
figure
mesh(xq,yq,zq_train_val);
hold on
plot3(train_val_X(1,:),train_val_X(2,:),train_val_Y,'.','MarkerSize',10,'linewidth',2);
title('Validation Data');
xlabel('X1');
ylabel('X2');
zlabel('Target (Tnew)')
legend('Surface','Scattered Points','Location','Best')
set(gca,'FontSize',20);


[xq,yq] = meshgrid(0:0.01:1, 0:0.01:1);
zq_test = griddata(test_X(1,:),test_X(2,:),test_Y,xq,yq);
figure
mesh(xq,yq,zq_test);
hold on
plot3(test_X(1,:),test_X(2,:),test_Y,'.','MarkerSize',10,'linewidth',2);
title('Test Data');
xlabel('X1');
ylabel('X2');
zlabel('Target (Tnew)')
legend('Surface','Scattered Points','Location','Best')
set(gca,'FontSize',20);

train_algos = {'traingd', 'traingda', 'traincgf', 'traincgp', 'trainbfg', 'trainlm', 'trainbr'}; 
hidden_sizes = [20 50 100];
transfer_funcs = {'logsig', 'tansig'};

net_final=feedforwardnet(50,'trainlm'); %hiddenSizes 
%(Row vector of one or more hidden layer sizes (default = 10)
%Row vector of one or more hidden layer sizes (default = 10), Training function

net_final.divideFcn = 'divideind';
net_final.divideParam.trainInd = train_val_idx;
net_final.layers{1}.transferFcn = char('tansig');

%training and simulation
net_final.trainParam.epochs=1000;  % set the number of epochs for the training 
net_final=train(net_final,train_val_X,train_val_Y);   % train the networks
% predictions on test data
pred_test=sim(net_final,test_X);  % simulate the networks with the input vector p
mse_test = mean((pred_test - test_Y).^2);

% bootstrap the mse on the test set
B = 10000;
[bootstat,bootsam] = bootstrp(B,[],test_X');
% transpose
bootsam_ = bootsam';
boot_store = cell(B,1);
for b=1:B   
    % select indices
    boot_test_X = test_X(:, bootsam_(b,:));
    boot_test_y = test_Y(bootsam_(b,:));
    % predict on bootstrapped test set
    boot_pred = sim(net_final,boot_test_X);
    % calcualte mse and store
    boot_mse = mean((boot_pred - boot_test_y).^2);
    boot_store{b} = boot_mse;
    
end;

% percentile confidence intrvals
bootstrap_ci_perc = prctile(cell2mat(boot_store),[2.5 97.5],1)

% The'bias-corrected and accelerated' (BCa) confidence interval;
compute_mse = @(x,y) mean((sim(net_final,x')-y').^2);
bootstrap_ci_bca = bootci(B,compute_mse,test_X', test_Y')
figure
hist(cell2mat(boot_store),50)
xlabel("Mean Squared Error")
title('Samples re-drawn from a single sample')
hold on
ylim = get(gca,'YLim');
h1=plot(bootstrap_ci_perc(1)*[1,1],ylim*1.05,'g-','LineWidth',2);
plot(bootstrap_ci_perc(2)*[1,1],ylim*1.05,'g-','LineWidth',2);
h2=plot(bootstrap_ci_bca(1)*[1,1],ylim*1.05,'r-','LineWidth',2);
plot(bootstrap_ci_bca(2)*[1,1],ylim*1.05,'r-','LineWidth',2);
legend([h1,h2],{'Percentile','Bca'});
hold off;
% save figure
% saveas(gcf,'output/fig9.png');

% visualize performance of chosen model on test data
% 1) plot surface
zq_test = griddata(test_X(1,:),test_X(2,:),test_Y,xq,yq);
figure
mesh(xq,yq,zq_test);
hold on
plot3(test_X(1,:),test_X(2,:),test_Y,'b.','MarkerSize',10,'linewidth',3);
hold on
plot3(test_X(1,:),test_X(2,:),pred_test,'r.','MarkerSize',15,'linewidth',3);

title('Performance of the best model on test data');
xlabel('X1');
ylabel('X2');
zlabel('Target')
legend('Surface ','Actual Points','Predicted Points','Location','Best')
set(gca,'FontSize',20);
hold off;