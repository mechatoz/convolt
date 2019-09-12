%% Convolutionl Dictionary Learning (CDL)
%| [Ref.] DOI: 10.1109/TIP.2017.2761545
%| Copyright 2019-09-10, Il Yong Chun, University of Hawaii
%| alpha ver 2018-03-05, Il Yong Chun, University of Michigan
clear; close all; clc;

%% Load and preprocess training data
addpath('../image_helpers');
preproc.CONTRAST_NORMALIZE = 'local_cn'; 
preproc.ZERO_MEAN = 1;
preproc.COLOR_IMAGES = 'gray';                         
[x] = CreateImages('../datasets/Images/fruit_100_100', ...
    preproc.CONTRAST_NORMALIZE, preproc.ZERO_MEAN, preproc.COLOR_IMAGES);
x = reshape(x, size(x,1), size(x,2), [] ); 


%% Parameters
%Number of block variables in CDL (default: 'multi')
param.blk = 'multi';    %option: 'multi', 'two'

%Hyperparameters
param.size_kernel = [11, 11, 100];    %the size and number of 2D filters
param.alpha = 1;                      %reg. param.: alpha in DOI: 10.1109/TIP.2017.2761545

%Majorization matrix options (default: {'D', 'D'})
%| Read descriptions of "Md_type" and "Mz_type" options in
%| the following functions: "BPEGM_CDL_2D_twoBlk.m" and "BPEGM_CDL_2D_multiBlk.m"
param.major_type = {'D', 'D'};        

%Options for BPEG-M algorithms
param.arcdegree = 95;    %param. for gradient-based restarting: angle between two vectors (degree)
param.max_it = 30;       %max number of iterations
param.tol = 1e-4;        %tol. val. for the relative difference stopping criterion

%Fixed random initialization
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);

%Initial filters and sparse codes
%| If [], then the CDL codes will automatically initialize them.
init.d = randn(param.size_kernel);    %no need to normalize
init.z = [];

%Display intermediate results?
verbose_disp = 1;    %option: 1, 0

%Save results?
saving = 1;   %option: 1, 0


%% CDL via BPEG-M
fprintf('CDL with %d x [%d x %d] kernels.\n\n', ...
    param.size_kernel(3), param.size_kernel(1), param.size_kernel(2) )

tic();
if strcmp(param.blk, 'two')
    prefix = 'BPEGM_CDL_twoBlock';
    [ d, z, Dz, obj, iterations] = BPEGM_CDL_2D_twoBlk(x, param.size_kernel, ...
        param.alpha, param.arcdegree, param.major_type(1), param.major_type(2), ...
        param.max_it, param.tol, verbose_disp, init);

elseif strcmp(param.blk, 'multi')
    prefix = 'BPEGM_CDL_multiBlock';
    [ d, z, Dz, obj, iterations ] = BPEGM_CDL_2D_multiBlk(x, param.size_kernel, ...
        param.alpha, param.arcdegree, param.major_type(1), param.major_type(2), ...
        param.max_it, param.tol, verbose_disp, init);
    
else
    error('The Block variable must be either of two or multi.');
end
tt = toc;

fprintf('CDL completed in %2.2f sec.\n\n', tt)


%% Show and save results
%Show learned filters
figure();    
pd = 1;
sqr_k = ceil(sqrt(size(d,3)));
d_disp = zeros( sqr_k * [param.size_kernel(1) + pd, param.size_kernel(2) + pd] + [pd, pd]);
for j = 0:size(d,3) - 1
    d_disp( floor(j/sqr_k) * (param.size_kernel(1) + pd) + pd + (1:param.size_kernel(1)),...
        mod(j,sqr_k) * (param.size_kernel(2) + pd) + pd + (1:param.size_kernel(2)) ) = d(:,:,j + 1); 
end
imagesc(d_disp); colormap gray; axis image;  colorbar; title('Final filter estimate');

%Save results
if saving == true
    save(sprintf('filters_%s_obj%3.3g.mat', prefix, obj), 'preproc', ...
        'param', 'init', 'd', 'z', 'Dz', 'obj', 'iterations');
    fprintf('Data saved.\n');
end
