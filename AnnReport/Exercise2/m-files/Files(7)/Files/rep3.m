%%%%%%%%%%%
% rep3.m
% A script which generates n random initial points for
% and visualise results of simulation of a 3d Hopfield network net
%%%%%%%%%%
T = [1 1 1; -1 -1 1; 1 -1 -1]';
net = newhop(T);
n=20;
for i=1:n
    a={rands(3,1)};                         % generate an initial point                   
    [y,Pf,Af] = sim(net,{1 50},{},a);       % simulation of the network  for 50 timesteps
    record=[cell2mat(a) cell2mat(y)];       % formatting results
    start=cell2mat(a);                      % formatting results 
    h = plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r','MarkerSize',20);  % plot evolution
    hold on;
    h_ = plot3(record(1,50),record(2,50),record(3,50),'gO','MarkerSize',20);  % plot the final point with a green circle
    set(h , 'linewidth',4);
    set(h_, 'linewidth',4);
end
grid on;
legend('initial state','time evolution','attractor','Location', 'Best');
title('Time evolution in the phase space of 3d Hopfield model');
xlabel('a(1)')
ylabel('a(2)')
zlabel('a(3)')
set(gca,'FontSize',24);

P = [-1.0 -0.5  0.0  +0.5   0.0   0  1   0.5   0.0   0.5  -0.5 -0.5;  
      0.0 -0.5  0.0  +0.5  +1.0  -1  0  -0.5  -0.5   0.0   0.5 -0.5;
      0.0 -0.5  0.0  +0.5   0.5   0  1   0.5   0.5  -0.5   0.5 -0.5];
figure
n = length(P);
for i=1:n
    a={P(:,i)};                         % generate an initial point                   
    [y,Pf,Af] = sim(net,{1 50},{},a);       % simulation of the network  for 50 timesteps
    record=[cell2mat(a) cell2mat(y)];       % formatting results
    start=cell2mat(a);                      % formatting results 
    h = plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r','MarkerSize',20);  % plot evolution
    hold on;
    h_ = plot3(record(1,50),record(2,50),record(3,50),'gO','MarkerSize',20);  % plot the final point with a green circle
    set(h , 'linewidth',4);
    set(h_, 'linewidth',4);
end
grid on;
legend('initial state','time evolution','attractor','Location', 'Best');
title('Time evolution in the phase space of 3d Hopfield model');
xlabel('a(1)')
ylabel('a(2)')
zlabel('a(3)')
set(gca,'FontSize',24);
