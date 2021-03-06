clear
clc
close all

%% load data and have a look at the first image
load threes -ascii;

colormap('gray');
imagesc(reshape(threes(1,:),16,16),[0,1]);

%% Plot the eigenvalues where D is the diagonal matrix of eigenvalues
% produces a diagonal matrix D of eigenvalues and 
% a full matrix V whose columns are the corresponding eigenvectors  
% so that A*V = V*D
C = cov(threes);
[V,D] = eig(C);

eigenvalues = flipud(diag(D));
eig_cumsum = cumsum(eigenvalues) / sum(eigenvalues);
plot(1 - eig_cumsum);
xlabel('q');
ylabel('quality loss');
xlim([0 255]);
ylim([0 1]);

%% compress data to 1,2,3,4 PC and visualize reconstruction of the images

% standardize first
avg = mean(threes);
sd = std(threes);
threes_stand = (threes - avg)./sd;
[coeff,score,latent,tsquared,explained] = pca(threes_stand);

% visualize variance explained in function of number of components
plot((explained),'linewidth',3)
xlim([1 size(explained,1)])
set(gca, 'YScale', 'log')
ylabel("Variance Exlained (log scale)")
xlabel("Number of Principal Components");

image_id = 15;

q = [10, 50, 150, 256];
figure
% add original plot
subplot(1,size(q,2)+1,1);
colormap('gray');
imagesc(reshape(threes(image_id,:),16,16),[0,1])
title("Original");
set(gca, 'visible', 'off')
set(findall(gca, 'type', 'text'), 'visible', 'on')

% reconstruct digits using only the 1,2,3,4 first components

colormap('gray');
for i = 1:size(q,2)
    subplot(1,size(q,2)+1,i+1);
	threes_hat = (score(:,1:q(i)) * coeff(:,1:q(i))') .* sd + avg;
    imagesc(reshape(threes_hat(image_id,:),16,16),[0,1]);
    title(['q = ', num2str(q(i))])
    set(gca, 'visible', 'off')
    set(findall(gca, 'type', 'text'), 'visible', 'on')
end
%%
q = [1:50];
rmse_q = [1:size(q,2)];
for i=q
    threes_hat = (score(:,1:i) * coeff(:,1:i)') .* sd + avg;
    rmse_q(i) = sqrt(mean(mean((threes-threes_hat).^2)));
end
figure
plot(rmse_q);
ylabel("RMSE")
xlabel("Number of Principal Components");

%%
C = cov(threes);
[V,D] = eig(C);
D_rotated = rot90(D,2)';
out_eig = [1:50];
for i=1:50
    out_eig(i) = D_rotated(i,i);
end

plot(rmse_q,'linewidth',4);
hold on;
plot(cumsum(out_eig),'r','linewidth',4)
set(gca,'FontSize',24)