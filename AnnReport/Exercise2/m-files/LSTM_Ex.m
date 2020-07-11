clc
clear all
close all
load lasertrain.dat
load laserpred.dat


data = [lasertrain.' laserpred.'];
figure,
subplot(2,1,1);
h1 = plot(lasertrain, 'MarkerSize',15);
xlabel('Time stamp');
title('Laser data set training data');
set(h1,'linewidth',3)
set(gca,'FontSize', 20)
subplot(2,1,2);
h2 = plot(laserpred);
xlabel('Time stamp');
title('Laser data set Prediction data');
set(h2,'linewidth',3)
set(gca,'FontSize', 20)

numTimeStepsTrain = 1000;

dataTrain = data(1:numTimeStepsTrain);
dataTest = data(numTimeStepsTrain:end);

mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 100;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',500, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',120, ...
    'LearnRateDropFactor',0.3, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

YPred = sig*YPred + mu;

YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2));

%Standardized MSE to compare to previous exercise
YPred = (YPred - mu)/sig;
mseTest = mean((YPred-dataTestStandardized(2:end)).^2);
YPred = sig*YPred + mu;

figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
h1 = plot(YTest);
hold on
h2 = plot(YPred,'.-');
hold off
set(h1,'linewidth',3)
set(h2,'linewidth',3)
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")
set(gca,'FontSize',20)

subplot(2,1,2)
h1 = stem(YPred - YTest)
set(h1,'linewidth',3)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)
set(gca,'FontSize',20)



net = resetState(net);
net = predictAndUpdateState(net,XTrain);

YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end

YPred = sig*YPred + mu;

rmse = sqrt(mean((YPred-YTest).^2));

figure
subplot(2,1,1)
h1 = plot(YTest);
hold on
h2 = plot(YPred,'.-');
hold off
set(h1,'linewidth',3)
set(h2,'linewidth',3)

legend(["Observed" "Predicted"])
ylabel("Cases")
title("Forecast with Updates")
set(gca,'FontSize',20)


subplot(2,1,2)
h_ = stem(YPred - YTest);
set(h_,'linewidth',3)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)
set(gca,'FontSize',20)


