%%kmeans完成图像的分割
function kmeans_demo()
clear;
close all;
clc;
im=imread('C:\\Users\\YAN\\Desktop\\kmeans_segment\\city.jpg');
imshow(im);
title('Input Image');
cform=makecform('srgb2lab');% create a color transformation structure
lab=applycform(im,cform);%应用到图片
ab=double(lab(:,:,2:3));
nrows=size(lab,1);
ncols=size(lab,2);
X=reshape(ab,nrows*ncols,2)';%ab大小没有变化
figure
scatter(X(1,:)',X(2,:)',3,'filled');%3决定marker的大小，filled this marker
box on;
%显示转换后的样本二维空间分布
%使用Kmeans进行聚类
k=5;
max_iter=100;%最大迭代次数
[centroids,labels]=run_kmeans(X,k,max_iter);
figure;%显示聚类分割的效果
scatter(X(1,:)',X(2,:)',3,labels,'filled');
hold on;
scatter(centroids(1,:),centroids(2,:),60,'r','filled');
hold on;
scatter(centroids(1,:),centroids(2,:),30,'g','filled');
box on;%adds a box to the current axes
hold off;
pixel_labels=reshape(labels,nrows,ncols);
rgb_labels=label2rgb(pixel_labels);%converts label matrix to RGB image.
figure
imshow(rgb_labels);
title('segmented Image');
end



% %%Kmeans 聚类算法的实现
function [centroids,labels]=run_kmeans(X,k,max_iter)
%输入：X样本集，K聚类中心个数，max_iter最大迭代次数
%输出：聚类中心，类标签
%采用Kmeans++初始化聚类中心
centroids=X(:,1+round(rand*(size(X,2)-1)));
labels=ones(1,size(X,2));
for i=2:k
    D=X-centroids(:,labels);
    D=cumsum(sqrt(dot(D,D,1)));%cumsum矩阵元素逐渐累加
    if D(end)==0
        centroids(:,i:k)=X(:,ones(k-i+1,1));
        return;
    end
    centroids(:,i)=X(:,find(rand<D/D(end),1));
    [~,labels]=max(bsxfun(@minus,2*real(centroids'*X),dot(centroids,centroids,1).'));
end
%Kmeans
for iter=1:max_iter
    for i=1:k
        l=labels==i;
        centroids(:,i)=sum(X(:,l),2)/sum(l);
    end
    [~,labels]=max(bsxfun(@minus,2*real(centroids'*X),dot(centroids,centroids,1).'),[],1);
end
end

















