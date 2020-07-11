clear
clc
close all

train_data = load('lasertrain.dat');

val_data = load('laserpred.dat');

% visualize data
subplot(2,1,1)
plot(train_data)
xlabel("Discrete time index")
ylabel("Amplitute")
title("Training Set")

subplot(2,1,2)
plot(val_data)
xlabel("Discrete time index")
ylabel("Amplitute")
title("Validation Set")

sizex = 20;
sizey = 20;
set(gcf, 'PaperPosition', [0 0 sizex sizey]);
set(gcf, 'PaperSize', [sizex sizey]);

train_data_trans = train_data;
mu_train = mean(train_data_trans);
sd_train = std(train_data_trans);
% standardize
train_stand = (train_data_trans - mu_train) / sd_train;


val_data_trans = val_data;
% standardize
val_stand = (val_data_trans - mu_train) / sd_train;

nr_lags = 20:70;
hidden_sizes = [10 50 100]; 
repeat_count = 3;
output = cell(size(nr_lags, 2) * size(hidden_sizes, 2) * repeat_count, 7);
batch = 0;         
counter = 1;

for lag=nr_lags
 
    X_train = getTimeSeriesTrainData(train_stand,lag);
    Y_train = train_stand(lag+1:end)';

    X_val = getTimeSeriesTrainData(([train_stand(end-(lag-1):end)',val_stand']'),lag);
    Y_val = val_stand';

    X_train_val = [X_train, X_val];
    Y_train_val = [Y_train, Y_val];

    for hidden=hidden_sizes
          batch = batch + 1;
          for j=1:repeat_count

                net_grid_search = feedforwardnet(hidden_sizes, 'trainscg');
                net_grid_search.trainParam.showWindow = false;
                net_grid_search.divideFcn = 'divideind';
               
                net_grid_search.trainParam.max_fail = 10; 
                net_grid_search.trainParam.epochs=500;

                net_grid_search.divideParam.trainInd = 1:size(X_train,2);
                
                net_grid_search.divideParam.valInd = size(X_train,2)+1:size(X_train,2)+size(X_val,2);
               
                tic;
                [net_grid_search, tr] = train(net_grid_search,X_train_val,Y_train_val, 'useGPU','yes'); 
                time = toc;

                val_X_window = train_stand(end-(lag-1):end);  
                Y_hat_val = sim(net_grid_search,val_X_window);
                % predictors
                if lag > 1
                    val_X_window = [val_X_window(2:end)',Y_hat_val]';
                % lags =  1
                else
                    val_X_window = Y_hat_val;
                end
                forecast_horizon = numel(val_data);
                for i = 2:forecast_horizon
                    % make predictions
                    Y_hat_val(i) = sim(net_grid_search, val_X_window);

                    if lag >1
                        val_X_window = [val_X_window(2:end)',Y_hat_val(i)]';
                    else
                        val_X_window = Y_hat_val(i);
                    end
                end

                Y_hat_val_orig_unit = sd_train*Y_hat_val + mu_train; 

                output{counter,1} = batch;
                output{counter,2} = tr.best_epoch;
                output{counter,3} = lag;
                output{counter,4} = hidden;
                output{counter,5} = mean((Y_hat_val_orig_unit-val_data').^2); %mse 
                output{counter,6} = sqrt(mean((Y_hat_val_orig_unit-val_data').^2)); % rmse
                output{counter,7} = time;
                counter = counter +1;
          end
    end
end

output_tbl = cell2table(output, 'VariableNames', {'batch','best_epoch','lag','hidden_size','mse_val',...
                 'rmse_val', 'time'});

          
group_stats  = grpstats(output_tbl, {'lag','hidden_size'}, {@median});            

figure
h1 = plot(nr_lags,group_stats.median_rmse_val(1:length(nr_lags)), 'bx-', 'MarkerSize',20);
hold on
set(h1,'linewidth',5);
h2 = plot(nr_lags,group_stats.median_rmse_val(length(nr_lags)+1:2*length(nr_lags)), 'rx-', 'MarkerSize',20);
hold on
set(h2,'linewidth',5);
h3 = plot(nr_lags,group_stats.median_rmse_val(2*length(nr_lags)+1:3*length(nr_lags)), 'gx-', 'MarkerSize',20);
set(h3,'linewidth',5)
hold on; xlabel('Lags');
ylabel('Error');
title('Performance of Time Series network')
set(gca,'FontSize',24)
legend('Hidden Units - 10','Hidden Units - 50','Hidden Units - 100','Location','Best');
set(gca,'FontSize',24)


figure
surf(hidden_sizes, nr_lags, reshape(group_stats.median_rmse_val,... 
     length(nr_lags),length(hidden_sizes)));
colorbar;
xlabel('Number of Lags');
ylabel('Size Hidden Layer');
zlabel('RMSE');
set(gca,'ZScale','log');
%caxis([0 15])

sizex = 20;
sizey = 20;
set(gcf, 'PaperPosition', [0 0 sizex sizey]);
set(gcf, 'PaperSize', [sizex sizey]);

lags = 40;
% train
X_train = getTimeSeriesTrainData(train_stand,lags);
Y_train = train_stand(lags+1:end)';

X_val = getTimeSeriesTrainData(([train_stand(end-(lags-1):end)',val_stand']'),lags);
Y_val = val_stand';

X_train_val = [X_train, X_val];
Y_train_val = [Y_train, Y_val];

% train network
net_train = feedforwardnet(30, 'trainlm');
net_train.trainParam.showWindow = true;
net_train.divideFcn = 'divideind';

net_train.trainParam.max_fail = 10; 
net_train.trainParam.epochs=70;

net_train.divideParam.trainInd = 1:size(X_train,2);

net_train.divideParam.valInd = size(X_train,2)+1:size(X_train,2)+size(X_val,2);

net_train = train(net_train,X_train_val,Y_train_val);   

val_X_window = train_stand(end-(lags-1):end);  
Y_hat_val = sim(net_train,val_X_window);
% predictors
if lags > 1
    val_X_window = [val_X_window(2:end)',Y_hat_val]';
% lags =  1
else
    val_X_window = Y_hat_val;
end
forecast_horizon = numel(val_data);
for i = 2:forecast_horizon

    Y_hat_val(i) = sim(net_train,val_X_window);

    if lags >1
        val_X_window = [val_X_window(2:end)',Y_hat_val(i)]';
    else
        val_X_window = Y_hat_val(i);
    end
end

Y_hat_val_orig_unit = sd_train*Y_hat_val + mu_train; 
mse_val = mean((Y_hat_val_orig_unit-val_data').^2);
rmse_val = sqrt(mean((Y_hat_val_orig_unit-val_data').^2));

figure
subplot(2,1,1)
plot(val_data)
hold on
plot(Y_hat_val_orig_unit,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Amplitute")
title("Forecast")

subplot(2,1,2)
stem(Y_hat_val_orig_unit - val_data')
xlabel("Discrete time index")
ylabel("Residuals")
title("RMSE = " + rmse_val)  

sizex = 20;
sizey = 20;
set(gcf, 'PaperPosition', [0 0 sizex sizey]);
set(gcf, 'PaperSize', [sizex sizey]);
