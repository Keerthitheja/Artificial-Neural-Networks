T = [+1 +1;
    -1 -1;
    1 -1];

net = newhop(T);

[Y,Pf,Af] = sim(net,2,[],T);
display(Y)
