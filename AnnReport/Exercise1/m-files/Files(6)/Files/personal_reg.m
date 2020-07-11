clear 
clc
close all

load Data_Problem1_regression.mat
Tnew=(8*T1+7*T2+5*T3+4*T4+3*T5)/(8+7+5+4+3);

perm=randperm(size(Tnew,1));
trainsety = Tnew(perm(1:100));
valsety =Tnew(perm(101:200));
testsety=Tnew(perm(201:300));

trainsetx1=X1(perm(1:100));
valsetx1=X1(perm(101:200));
testsetx1=X1(perm(201:300));

trainsetx2=X2(perm(1:100));
valsetx2=X2(perm(101:200));
testsetx2= X2(perm(201:300));

%figure
F=scatteredInterpolant(trainsetx1,trainsetx2,trainsety);
[xq,yq]=meshgrid(0:0.01:1);
F.Method='nearest';
vq1=F(xq,yq);
mesh(xq,yq,vq1);

p=con2seq(transpose([trainsetx1,trainsetx2]));
t=con2seq(transpose(trainsety));

valp=con2seq(transpose([valsetx1,valsetx2]));
valt=con2seq(transpose(valsety));
net1=feedforwardnet([100,20],'trainlm');
net1=train(net1,p,t);