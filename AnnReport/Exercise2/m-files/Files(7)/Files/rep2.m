%%%%%%%%%%%
% rep2.m
% A script which generates n random initial points 
%and visualises results of simulation of a 2d Hopfield network 'net'
%%%%%%%%%%

T = [1 1; -1 -1; 1 -1]';
net = newhop(T);
n=20;
num_steps = 50;
for i=1:n
    a={rands(2,1)};                     % generate an initial point 
    %a = {[0 ;0]}
    [y,Pf,Af] = sim(net,{1 num_steps},{},a);   % simulation of the network for 50 timesteps              
    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    start=cell2mat(a);                  % formatting results 
    h = plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r', 'MarkerSize', 20); % plot evolution
    hold on;
    h_ = plot(record(1,num_steps),record(2,num_steps),'gO','MarkerSize',20);  % plot the final point with a green circle
    set(h , 'linewidth',4);
    set(h_, 'linewidth',4);
end

set(gca,'FontSize',24);
legend('initial state','time evolution','attractor','Location','Best');
title('Time evolution in the phase space of 2d Hopfield model');

P = [-1.0 -0.5 0.0 +0.5 0 0 1 0.5 0 0.5 -0.5;  
     0 -0.5 0.0 +0.5 +1.0 -1 0 -0.5 -0.5 0 0.5];
figure
 for i=1:length(P)
    a={P(:,i)};                     % generate an initial point 
    %a = {[0 ;0]}
    [y,Pf,Af] = sim(net,{1 num_steps},{},a);   % simulation of the network for 50 timesteps              
    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    start=cell2mat(a);                  % formatting results 
    h = plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r', 'MarkerSize', 20); % plot evolution
    hold on;
    h_ = plot(record(1,num_steps),record(2,num_steps),'gO','MarkerSize',20);  % plot the final point with a green circle
    set(h , 'linewidth',4);
    set(h_, 'linewidth',4);
end

set(gca,'FontSize',24);
legend('initial state','time evolution','attractor','Location','Best');
title('Time evolution in the phase space of 2d Hopfield model');