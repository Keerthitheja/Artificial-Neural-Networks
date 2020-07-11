clear
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'traincgf' and 'trainbr'
% trainbr - batch gradient descent 
% traincgf - Levenberg - Marquardt
%%%%%%%%%%%

%generation of examples and targets
sigma=0.5;
%generation of examples and targets
x=0:0.05:3*pi; y=sin(x.^2);%+sigma*randn(1,length(x));
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

train_algos = {'traingd', 'traincgf','trainlm', 'trainbr'}; 
hidden_sizes = [30 50 100 150];

counter = 1;
batch = 0;
repeat = 1;
for hidden_size=hidden_sizes
    for train_algo=train_algos
            batch = batch + 1;   
            for j = 1:repeat
                net_train = feedforwardnet(hidden_size, char(train_algo));
                net_train.trainParam.epochs = 1000;
                net_train.trainParam.showWindow = true;
                net_train.trainParam.goal = 0;
                % this is to ensure we obtain the mse for the trainbr on
                % the validation data (is disabled by default)
                net_train.trainParam.max_fail = 20;                 
                % time training of the network
                tic;
                [net_train, tr] = train(net_train,p,t);   
                time = toc;
                  
                data{counter, 1} = train_algo;
                data{counter, 2} = batch;
                data{counter, 3} = hidden_size;               
                data{counter, 4} = tr.best_epoch;
                data{counter, 5} = tr.best_perf;
                data{counter, 6} = time;

                counter = counter + 1;
                
                % print every 10 iterations
                if mod(counter,10)==1
                    fprintf('%3i\n', counter-1)
                end   
            end
    end
end

tbl = cell2table(data, 'VariableNames', {'train_algos', 'repetition', 'hidden_size',...
                 'best_epoch','mse_train','time'});
display(tbl);