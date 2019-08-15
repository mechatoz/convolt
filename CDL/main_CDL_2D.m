%% Convolutionl Dictionary Learning (CDL)
%| Copyright 2019-08-13, Il Yong Chun, University of Hawaii
%| alpha ver 2018-03-05, Il Yong Chun, University of Michigan
%| Ref. DOI: 10.1109/TIP.2017.2761545
clear; close all; clc;

%% Load and preprocess training data
addpath('../image_helpers');
preproc.CONTRAST_NORMALIZE = 'local_cn'; 
preproc.ZERO_MEAN = 1;
preproc.COLOR_IMAGES = 'gray';                         
[b] = CreateImages('../datasets/Images/fruit_100_100', ...
    preproc.CONTRAST_NORMALIZE, preproc.ZERO_MEAN, preproc.COLOR_IMAGES);
b = reshape(b, size(b,1), size(b,2), [] ); 


%% Parameters
%Number of block variables in CDL (default: 'multi')
blk = 'multi';    %option: 'multi', 'two'

%Hyperparameters
kernel_size = [11, 11, 100];    %the size and number of 2D filters
alpha = 1;                      %reg. param.: alpha in DOI: 10.1109/TIP.2017.2761545

%Majorization matrix options (default: {'D', 'D'})
%| Read descriptions of "Md_type" and "Mz_type" options in
%| the following functions: "BPEGM_CDL_2D_twoBlk.m" and "BPEGM_CDL_2D_multiBlk.m"
major_type = {'D', 'D'};        

%Options for BPEG-M algorithms
verbose_disp = 1;    %option: 1, 0
max_it = 30;
tol = 1e-4;

%Fixed random initialization
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);

%Initial filters and sparse codes
%| If [], then the CDL codes will automatically initialize them.
init.d = randn(kernel_size);    %no need to normalize
init.z = [];

%Save results?
saving = 1;   %option: 1, 0


%% CDL
fprintf('CDL with %d x [%d x %d] kernels.\n\n', ...
    kernel_size(3), kernel_size(1), kernel_size(2) )

tic();
if strcmp(blk, 'two')
    prefix = 'BPEGM_CDL_twoBlock';
    [ d, z, Dz, obj, iterations]  = BPEGM_CDL_2D_twoBlk(b, kernel_size, alpha, ...
        major_type(1), major_type(2), max_it, tol, verbose_disp, init);

elseif strcmp(blk, 'multi')
    prefix = 'BPEGM_CDL_multiBlock';
    [ d, z, Dz, obj, iterations ]  = BPEGM_CDL_2D_multiBlk(b, kernel_size, alpha, ...
        major_type(1), major_type(2), max_it, tol, verbose_disp, init);
    
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
d_disp = zeros( sqr_k * [kernel_size(1) + pd, kernel_size(2) + pd] + [pd, pd]);
for j = 0:size(d,3) - 1
    d_disp( floor(j/sqr_k) * (kernel_size(1) + pd) + pd + (1:kernel_size(1)) , mod(j,sqr_k) * (kernel_size(2) + pd) + pd + (1:kernel_size(2)) ) = d(:,:,j + 1); 
end
imagesc(d_disp); colormap gray; axis image;  colorbar; title('Final filter estimate');

%Save results
if saving == true
    save(sprintf('filters_%s_obj%3.3g.mat', prefix, obj), 'preproc', ...
        'd', 'z', 'Dz', 'obj', 'iterations', 'alpha', 'major_type', ...
        'max_it', 'tol');
end