clc
clear all
close all

N = 500;
n_dim = 50;
qList = 1:n_dim ;
X = randn(n_dim, N);
avg = mean(mean(X));
X_norm = X - avg;
for q = qList
    [E, d_cumquality] = mypca(X_norm, q);
    F = E;
    quality = d_cumquality(q);
    Z = E'*X_norm;
    X_hat = F*Z + avg;
    mserror(q) = sqrt(mean(mean((X-X_hat).^2)));
end
% figure
% plot(E,mserror);

load choles_all
X = p;
[n_dim , n_obs] = size(X);
q = n_dim;
avg = mean(mean(X));
X_norm = X - avg;
for i = 1:q
    [E, d_cumquality] = mypca(X_norm, i);
    F = E;
    quality = d_cumquality(i);
    Z_norm = E'*X_norm;
    X_hat = F*Z_norm + avg;
    rmse(i) = sqrt(mean(mean((X-X_hat).^2)));
end

figure
plot(qList,mserror,'b*-','Markersize',15,'linewidth',3);
hold on
plot(1:q,rmse,'r*-','Markersize',15,'linewidth',3);
xlabel('-- Number of Eigen Values --');
ylabel('RMSE');
title('PCA ; Reconstruction errors over number of eigen values for Random Gaussian and Choles data');
legend('Random Gaussian Data','Choles Data','location','Best');
set(gca,'FontSize',20);