%% EXAMPLE OF PERCEPTRON
%training data
clc, clear; 
%Example 1
% X = [ -0.5 -0.5 +0.3 -0.1;  ...
%       -0.5 +0.5 -0.5 +1.0];
% T = [1 1 0 0];
%Example 4
% X = [-0.5 -0.5 +0.3 -0.1 -40; -0.5 +0.5 -0.5 +1.0 50];
% T = [1 1 0 0 1];
%Example 5
X = [ -0.5 -0.5 +0.3 -0.1 -40; ...
      -0.5 +0.5 -0.5 +1.0 50];
T = [1 1 0 0 1];
%Example 6
% X = [ -0.5 -0.5 +0.3 -0.1 -0.8; ...
%       -0.5 +0.5 -0.5 +1.0 +0.0 ];
% T = [1 1 0 0 0];



% X = [-2 +0.5 +1.5 -1; -1.5 -1 -2 +2]
% T = [1 0 0 0]    
% 
% X = [-0.5 -0.5 +0.3 -0.1 -40; -0.5 +0.5 -0.5 +1.0 50];
% T = [1 1 0 0 1];
figure
plotpv(X,T); %Draw the points that we have in our training data

%% INITIALIZATION OF PERCEPTRON

net = perceptron('hardlim','learnp');
net = configure(net,X,T);
hold on
linehandle = plotpc(net.IW{1},net.b{1}); %Weights plot

%% learninng
E = 1;
epoch = 0;
while (sse(E)&&epoch <25)
    epoch=epoch+1
   [net,Y,E] = adapt(net,X,T);%On line learning
   linehandle = plotpc(net.IW{1},net.b{1},linehandle); %Variable with the plots of weights and bias of each epoch
   drawnow;
end


%% Predictions-test

x = [0.7; 1.2];
y = net(x)
plotpv(x,y);
circle = findobj(gca,'type','line');
circle.Color = 'red';
%Complete draw
hold on;
plotpv(X,T);
plotpc(net.IW{1},net.b{1});
hold off;