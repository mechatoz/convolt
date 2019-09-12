%% Convolutional Analysis Operator Learning (CAOL)
%| [Ref.] DOI: 10.1109/TIP.2019.2937734
%| Copyright 2019-09-10, Il Yong Chun, University of Hawaii
%| alpha ver 2018-03-05, Il Yong Chun, University of Michigan
clear; close all; clc;

%% Load and preprocess training data
addpath('../image_helpers');
preproc.CONTRAST_NORMALIZE = 'local_cn'; 
preproc.ZERO_MEAN = 1;
preproc.COLOR_IMAGES = 'gray';       
[x] = CreateImages('../datasets/Images/city_100_100', ...
    preproc.CONTRAST_NORMALIZE, preproc.ZERO_MEAN, preproc.COLOR_IMAGES);
x = reshape(x, size(x,1), size(x,2), [] ); 

addpath('../HNO');

%% Parameters
%Type of filter regularizers in CAOL (default: 'tf')
param.reg = 'tf';    %option: 'tf', 'div'

%Hyperparameters
param.size_kernel = [7, 7, 49];   %the size and number of 2D filters
param.alpha = 1e-3;               %reg. param.: alpha in DOI: 10.1109/TIP.2019.2937734
                            %(for cont. enh. & mean subtract, default: [1e-3, 5e-3])
                            %(for mean subtract, default: 2.5×{e-5, e-4})

%Majorization matrix options (default: 'H')
%| Read descriptions of "M_type" in the following function: "BPEGM_CAOL_2D_TF.m" 
param.major_type = 'H';   

%Options for BPEG-M algorithms
param.lambda = 1+eps;    %scaling param. for majorization matrix (default: 1+eps)
param.arcdegree = 90;    %param. for gradient-based restarting: angle between two vectors 
                   %(default: 90 degree)
param.max_it = 1e3;      %max number of iterations
param.tol = 1e-4;        %tol. val. for the relative difference stopping criterion

%Fixed random initialization
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);

%Initial filters
%| If [], then the CDL codes will automatically initialize them.
init_d = randn(param.size_kernel);    %no need to normalize

%Display intermediate results?
verbose_disp = 1;    %option: 1, 0

%Save results?
saving = 1;   %option: 1, 0


%% CAOL via BPEG-M
fprintf('CAOL with %d x [%d x %d] kernels.\n\n', ...
    param.size_kernel(3), param.size_kernel(1), param.size_kernel(2) )

tic();
if strcmp(param.reg, 'tf')
    prefix = 'CAOL_TF';
    [ d, z, x_filt, obj, iterations ] = BPEGM_CAOL_2D_TF(x, param.size_kernel, ...
        param.alpha, param.lambda, param.arcdegree, param.major_type, ...
        param.max_it, param.tol, verbose_disp, init_d);

elseif strcmp(param.reg, 'div')
    prefix = 'CAOL_div';
    error('This subroutine will be added soon...');

else
    error('The filter regularizer must be either of TF const. or diversity-promoting reg.');
end
tt = toc;

fprintf('CAOL completed in %2.2f sec.\n\n', tt);


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
        'param', 'init_d', 'd', 'z', 'x_filt', 'obj', 'iterations');
    fprintf('Data saved.\n');
end
