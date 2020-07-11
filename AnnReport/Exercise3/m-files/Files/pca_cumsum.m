clc; clear; close all
load threes -ascii

figure
colormap('gray');
imagesc(reshape(threes(2,:),16,16),[0,1])
%saveas(gcf,'Orignal2.jpg');
close all
figure
colormap('gray');
imagesc(reshape(threes(10,:),16,16),[0,1])
%saveas(gcf,'Orignal10.jpg');
close all
figure
colormap('gray');
imagesc(reshape(threes(50,:),16,16),[0,1])
%saveas(gcf,'Orignal50.jpg');
close all

avg = mean(threes,1)
covMatrix = cov(threes);

kVectors = [256]; 
[rows, cols] = size(kVectors); 
[v1,d1] = eig(covMatrix);
eigenValues = diag(d1)';
newVector = zeros(size((eigenValues)));

for i=1:256
    newVector(end-i+1) = eigenValues(i)
end
cummulativeEigneValue= cumsum(newVector); 
%bar(newVector)
title('Eigenvalues')
xlabel('Eigenvalues')
ylabel('Amount of information')
%bar(diag(d1));
errorVector = [];
for q=1:50
    
    %bar(diag(d1));
    zeroMeanX = (threes-avg)';
    [v,d] = eigs(covMatrix,(q));
    E = v;
    
    z = (E'*zeroMeanX);
    xhat = (E*z)';

    xhat = xhat+avg;
    
    error = (sqrt(mean(mean((threes-xhat).^2))));
    errorVector = [errorVector;error];

%     figure
%     colormap('gray');
%     imagesc(reshape(xhat(2,:),16,16),[0,1])
%     saveas(gcf,strcat('img2_',int2str(q),'.jpg'));
%     close all
%     figure
%     colormap('gray');
%     imagesc(reshape(xhat(10,:),16,16),[0,1])
%     saveas(gcf,strcat('img10_',int2str(q),'.jpg'));
%     close all
%     figure
%     colormap('gray');
%     imagesc(reshape(xhat(50,:),16,16),[0,1])
%     saveas(gcf,strcat('img50_',int2str(q),'.jpg'));
%     close all
end
%figure
%bar(diag(d1));
figure
plot(1:50, errorVector/errorVector(1),'b', 'linewidth',3);
hold on
plot(1:50, cummulativeEigneValue(1:50)/20.3776,'r','linewidth',3);
grid off
legend('Reconstruction error', 'Eigenvalues information');
title('RMSE vs Eigenvalues');
xlabel('Number of eigenvalues')
ylabel('RMSE & Eigenvalues information');
set(gca,'FontSize',20);
% saveas(gcf,strcat('ErrorVSEigenvalues.jpg'));

%imagesc(reshape(xhat(2,:),16,16),[0,1])