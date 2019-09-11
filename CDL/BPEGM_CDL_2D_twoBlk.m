function [ d_res, z_res, Dz, obj_val, iterations ] = ...
        BPEGM_CDL_2D_twoBlk( x, size_kernel, alpha, Md_type, Mz_type, ...
        max_it, tol, verbose, init )
    
%| BPEGM_CDL_2D_twoBlk:
%| Two-block ver. Convolutional Dictionary Learning (CDL) via 
%| Block Proximal Extrapolated Gradient method using Majorization and
%| gradient-based restarting scheme (reG-BPEG-M, block multi-convex ver.)
%|
%| [Input]
%| x: training images in sqrt(N) x sqrt(N) x L
%| size_kernel: [psf_s, psf_s, K]
%| alpha: reg. param. for sparsifying regularizer (l1 term)
%| Md_type: majorization matrix option for filter update -- 'I','D','Dtgt'
%|         'D' is Lem. 5.1 in DOI: 10.1109/TIP.2017.2761545
%| Mz_type: majorizaiton matrix option for sparse code update -- 'D','Dtgt'
%|         'D' is Lem. 5.2 in DOI: 10.1109/TIP.2017.2761545
%| max_it: max number of iterations
%| tol: tolerance value for the relative difference stopping criterion
%| verbose: option to show updated filters
%| init: initial values for filters, sparse codes
%|
%| [Output]
%| d_res: learned filters in [psf_s, psf_s, K]
%| z_res: final updates of sparse codes
%| Dz: final synthesized images
%| obj_val: final objective value
%| iterations: records for BPEG-M iterations 
%|
%| Copyright 2019-09-10, Il Yong Chun, University of Hawaii
%| alpha ver 2018-03-05, Il Yong Chun, University of Michigan
           

%% Def: Parameters, Variables, and Operators
psf_s = size_kernel(1);
K = size_kernel(3);
L = size(x,3);

%variables for filters
center = floor([size_kernel(1), size_kernel(2)]/2) + 1;
psf_radius = floor( psf_s/2 );

%variable dimensions
size_xpad = [size(x,1) + psf_s-1, size(x,2) + psf_s-1, L];
size_z = [size_xpad(1), size_xpad(2), K, L];

%Objective
objective = @(z, dh) objectiveFunction( z, dh, x, alpha, ...
    psf_radius, center, psf_s, size_z, size_xpad );

%Operator for padding/unpadding to filters
PS = @(u) dpad_to_d(u, center, size_kernel);
PSt = @(u) d_to_dpad(u, size_xpad, size_kernel, center);

%Proximal operator for l1 norm
ProxSparseL1 = @(u, a) sign(u) .* max( abs(u)-a, 0 );

%Mask and padded data
[PBtPB, PBtx] = pad_data(x, center(1), psf_s);

%Adaptive restarting: Cos(ang), ang: the angle between two vectors
omega = cos(pi*95/180);   


%% Initialization
%Initialization: filters
if ~isempty(init.d)
    d = init.d;
else
    %Random initialization
    d = randn(size_kernel);
end
%filter normalization
dnorm = @(x) bsxfun(@rdivide, x, sqrt(sum(sum(x.^2, 1), 2)));
d = dnorm(d);
d_hat = fft2( PSt(d) );
d_p = d;

%Initialization: sparse codes
if ~isempty(init.z)
    z = init.z;
else
    %Random initialization (starts with higher obj. value)
    %z = randn(size_z);
    
    %Fixed initialization
    z = zeros(size_z);
    for l = 1:L
        for k = 1:K
            %circular padding
            if mod(psf_s,2) == 0
                xpad = padarray(x(:,:,l), [center(1)-1, center(2)-1], 'circular', 'both');    %circular pad
                xpad = xpad(1:end-1, 1:end-1, :, :);
            else
                xpad = padarray(x(:,:,l), [center(1)-1, center(2)-1], 'circular', 'both');    %circular pad
            end 
            z(:,:,k,l) = ProxSparseL1(xpad, alpha) / K;
        end
    end
end
z_hat = reshape( fft2( reshape(z, size_z(1), size_z(2), []) ), size_z );
z_p = z;

%ETC
tau_d = 1;          %momentum coeff. for filters
tau_z = 1;          %momentum coeff. for sparse codes
weight = 1-eps;     %delta in Prop. 3.2 of DOI: 10.1109/TIP.2017.2761545

%Save all objective values and timings
iterations.obj_vals_d = [];
iterations.obj_vals_z = [];
iterations.tim_vals = [];
iterations.it_vals = [];

%Initial vals
obj_val = objective(z, d_hat);

%Save all initial vars
iterations.obj_vals_d(1) = obj_val;
iterations.obj_vals_z(1) = obj_val;
iterations.tim_vals(1) = 0;
iterations.it_vals = cat(4, iterations.it_vals, d );

%Debug progress
fprintf('Iter %d, Obj %3.3g, Diff %5.5g\n', 0, obj_val, 0)

%Display filters
if verbose == 1
    iterate_fig = figure();
    filter_fig = figure();
    display_func(iterate_fig, filter_fig, d, d_hat, z_hat, x, size_xpad, size_z, psf_radius, 0);
end

  
%% %%%%%%%%%% Two-block CDL via reG-BPEG-M %%%%%%%%%%
for i = 1:max_it

    %% UPDATE: All filters, { d_k : k=1,...,K }

    %Compute majorization matrices
    tic; %timing
    if i==1
       Md = maj_for_d(z_hat, Md_type, size_z, size_kernel, PS, PSt); 
    else
       Md_old = Md;
       Md = maj_for_d(z_hat, Md_type, size_z, size_kernel, PS, PSt); 
    end
    t_kernel_maj = toc; %timing

    %Proximal operator for (ineq.) unit-norm constraint
    ProxKernelConstraint = @(u) KernelConstraintProj( u, Md, Md_type );

    %System operators
    A = @(u) A_for_d( z_hat, PBtPB, size_z, PSt, u );      
    Ah = @(u) Ah_for_d( conj(z_hat), size_z, PS, u );

    tic; %timing

    %%%%%%%%%%%%%%%%%%%%% reG-BPEG-M %%%%%%%%%%%%%%%%%%%%%%             
    if i ~= 1              
        %Extrapolation with momentum!
        E_d = weight * min( repmat( (tau_d_old - 1)/tau_d, size(Md) ), sqrt( Md_old ./ Md ) );  
        d_p = d + E_d .* (d - d_old);
    end

    %Proximal mapping
    d_old = d;
    d = ProxKernelConstraint( d_p - Ah( A(d_p) - PBtx ) ./ Md );

    %Gradient-based adaptive restarting
    Md_diff = Md .* (d-d_old);
    if dot( d_p(:)-d(:), Md_diff(:) ) / ( norm(d_p(:)-d(:)) * norm(Md_diff(:)) ) > omega
        d_p = d_old;
        d = ProxKernelConstraint( d_p - Ah( A(d_p) - PBtx ) ./ Md );
        disp('Restarted!');
    end
    
    %Momentum coeff. update
    tau_d_old = tau_d;
    tau_d = ( 1 + sqrt(1 + 4*tau_d^2) ) / 2;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    d_hat = fft2( PSt( d ) );
    
    %timing
    t_kernel_bpegm = toc;
    t_kernel = t_kernel_maj+ t_kernel_bpegm;
    
    
    %% EVALUATION
    %Debug progress
    obj_val = objective(z, d_hat);  
    d_relErr = norm(d(:)-d_old(:),2)/norm(d(:),2);        
    fprintf('Iter %d D, Obj %3.3g, Diff %5.5g\n', i, obj_val, d_relErr);
    
    %Record current iteration
    iterations.obj_vals_d(i + 1) = obj_val;
    

    %% UPDATE: All sparse codes, { z_{l,k} : l=1,...,L, k=1,...,K }

    %Compute majorization matrices
    tic; %timing   
    if i==1
       Mz = maj_for_z(d_hat, Mz_type, size_z); 
    else
       Mz_old = Mz;
       Mz = maj_for_z(d_hat, Mz_type, size_z); 
    end
    t_spcd_maj = toc; %timing

    %System operators
    A = @(u) A_for_z( d_hat, PBtPB, size_z, u );      
    Ah = @(u) Ah_for_z( conj(d_hat), size_z, u );   

    tic; %timing

    %%%%%%%%%%%%%%%%%%%%% reG-BPEG-M %%%%%%%%%%%%%%%%%%%%%%                  
    if i ~= 1
        %Extrapolation with momentum!
        E_z = weight * min( repmat( (tau_z_old - 1)/tau_z, size(Mz) ), ...
            sqrt( Mz_old ./ Mz ) );
        z_p = z + E_z .* (z - z_old); 
    end

    %Proximal mapping
    z_old = z;
    z = ProxSparseL1( z_p - Ah( A(z_p) - PBtx ) ./ Mz, alpha ./ Mz );

    %Gradient-based adaptive restarting
    Mz_diff = Mz .* (z-z_old);
    if dot( z_p(:)-z(:), Mz_diff(:) ) / ( norm(z_p(:)-z(:)) * norm(Mz_diff(:)) ) > omega
        z_p = z_old;
        z = ProxSparseL1( z_p - Ah( A(z_p) - PBtx ) ./ Mz, alpha ./ Mz );
        disp('Restarted!');
    end

    %Momentum coeff. update
    tau_z_old = tau_z;
    tau_z = ( 1 + sqrt(1 + 4*tau_z^2) ) / 2;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    z_hat = reshape( fft2( reshape(z, size_xpad(1), size_xpad(2),[]) ), size_z );
    
    %timing    
    t_spcd_bpegm = toc;
    t_spcd = t_spcd_maj + t_spcd_bpegm;
    
    
    %% EVALUATION
    %Debug progress
    obj_val = objective(z, d_hat);
    z_relErr = norm(z(:)-z_old(:),2)/norm(z(:),2);
    fprintf('Iter %d Z, Obj %3.3g, Diff %5.5g\n', i, obj_val, z_relErr)
    
    %Record current iteration
    iterations.obj_vals_z(i + 1) = obj_val;
    iterations.tim_vals(i + 1) = iterations.tim_vals(i) + t_kernel + t_spcd;
    if mod(i,100) == 0  %save filters every 100 iteration
        iterations.d = cat(4, iterations.d, d );
    end
    
    %Display filters
    if verbose == 1
        if mod(i,25) == 0
            display_func(iterate_fig, filter_fig, d, d_hat, z_hat, b, size_xpad, size_z, psf_radius, i);
        end
    end
    
    %Termination
    if d_relErr < tol && z_relErr < tol
         disp('relErr reached');      
        break;
    end
    
end

%Final estimate
z_res = z;
d_res = d;
Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,L]) .* z_hat, 3), size_xpad) ));

return;





%%%%%%%%%%%%%%%%%%%% Def: Padding Operators %%%%%%%%%%%%%%%%%%%%

function d = dpad_to_d(dpad, center, size_kernel)

%pad to filters (for multiple filters)
d = circshift(dpad, -[1-center(1), 1-center(2), 0]);
d = d( 1:size_kernel(1), 1:size_kernel(2), : );

return;


function dpad = d_to_dpad(d, size_xpad, size_kernel, center)

%remove padding from filters (for multiple filters)
dpad = padarray( d, [size_xpad(1)-size_kernel(1), size_xpad(2)-size_kernel(2), 0], 0, 'post' );
dpad = circshift(dpad, [1-center(1), 1-center(2), 0]);

return;


function [MtM, Mtx] = pad_data(x, center, psf_s)

if mod(psf_s,2) == 0
    MtM = padarray(ones(size(x)), [center-1, center-1, 0], 0, 'both');    %mask
    MtM = MtM(1:end-1, 1:end-1, :);
    Mtx = padarray(x, [center-1, center-1, 0], 0, 'both');    %padded x
    Mtx = Mtx(1:end-1, 1:end-1, :);
else
    MtM = padarray(ones(size(x)), [center-1, center-1, 0], 0, 'both');    %mask
    Mtx = padarray(x, [center-1, center-1, 0], 0, 'both');    %padded x
end
        
return;




%%%%%%%%%%%%%%%%%%%% Def: System Operators %%%%%%%%%%%%%%%%%%%%

function Axi = A_for_d( z_hat, mask, size_z, PSt, xi )

sy = size_z(1); sx = size_z(2); K = size_z(3); L = size_z(4);   %size

Axi = real( ifft2( reshape( sum( repmat( fft2( PSt( xi ) ), [1,1,1,L] ) .* z_hat, 3 ), [sy, sx, L] ) ) );
Axi(~mask) = 0;     %consider PB'PB

return;


function Ahxi = Ah_for_d( zH_hat, size_z, PS, xi )

sy = size_z(1); sx = size_z(2); K = size_z(3); L = size_z(4);   %size

Ahxi = PS( real( ifft2( reshape( sum( permute( repmat( fft2( xi ), [1,1,1,K] ), [1 2 4 3] ) .* zH_hat, 4 ), [sy, sx, K] ) ) ) );

return;


function Axi = A_for_z( d_hat, mask, size_z, xi )

sy = size_z(1); sx = size_z(2); K = size_z(3); L = size_z(4);   %size

Axi = real(ifft2( reshape( sum( fft2(xi) .* repmat(d_hat,[1,1,1,L]), 3 ), [sy, sx, L] ) ));
Axi(~mask) = 0;     %consider PB'PB

return;


function Ahxi = Ah_for_z( dH_hat, size_z, xi )

sy = size_z(1); sx = size_z(2); K = size_z(3); L = size_z(4);   %size

xi_hat_mat = permute( repmat( fft2( xi ), [1,1,1,K] ), [1 2 4 3] );
dH_hat_mat = repmat( dH_hat, [1,1,1,L] );
Ahxi = reshape( real( ifft2( reshape( xi_hat_mat .* dH_hat_mat, sy, sx,[] ) ) ), size_z );

return;




%%%%%%%%%%%%%%%%%%%% Design: Majorization Matrices %%%%%%%%%%%%%%%%%%%%

function Md = maj_for_d(z_hat, Md_type, size_z, size_kernel, PS, PSt)
% Compute majorizer for filter updates

sy = size_z(1); sx = size_z(2); K = size_z(3); L = size_z(4);   %size

%Permute spectra: each cell corresponds to frequency
zhat_mat = reshape( num2cell( permute( reshape(z_hat, [sy*sx, K, L] ), [3,2,1] ), [1 2] ), [1 sy*sx]); %n * k * s

if strcmp( Md_type, 'I')
    %Scaled identity majorization matrix in
    %Lem. 4.3 of DOI: 10.1109/TIP.2017.2761545
    
    %construct { Sigma_k : k=1,...,D }
    ZhatHZhat = cellfun(@(A)(sum( abs(A'*A), 2 )), zhat_mat, 'UniformOutput', false');
    
    %reshape to N'-by-D
    d_maj_freq = permute(cell2mat(ZhatHZhat), [2,1]);
    d_maj = max(d_maj_freq);
    
    %matrix construction    
    Md(1,1,:) = d_maj;
    Md = repmat(Md, [size_kernel(1) size_kernel(1) 1]);
    
elseif strcmp( Md_type, 'D')
    %Diagonal majorization matrix in
    %Lem. 4.4 of DOI: 10.1109/TIP.2017.2761545
    
    %construct { Sigma_k : k=1,...,D }
    ZhatHZhat = cellfun(@(A)(sum( abs(A'*A), 2 )), zhat_mat, 'UniformOutput', false');
    
    %reshape to sqrt(N')-by-sqrt(N')-by-D
    d_maj_freq = reshape( permute(cell2mat(ZhatHZhat), [2,1]), [sy,sx,K] );
    %d_maj_freq = reshape( permute(vertcat(ZhatHZhat{:}), [2,1]), [sy,sx,K] );    %slightly faster

    %|F' Sigma_k F| = circulant matrix with absolute values only
    d_maj_freq = fft2( abs( ifft2(d_maj_freq) ) );
    
    %P_S |F' Sigma_k F| P'_S * 1vec
    Md = PS( real( ifft2( d_maj_freq .* fft2( PSt( ones(size_kernel) ) ) ) ) );
    
elseif strcmp( Md_type, 'Dtgt')
    %Diagonal majorization matrix in
    %Lem. 4.1 of DOI: 10.1109/TIP.2017.2761545
    %| Note: Compared to 'D1', 'D2' obtains sharper majorization
    %|       (i.e., faster convgergence), but requires more compt.
    
    %reshape to sqrt(N')-by-sqrt(N') freq-cell:
    %ith columns of cells correspond to rows of [ Zhat' Zhat ]_{i,*}
    ZhatHZhat = cellfun(@(A)( ( A'*A ).' ), zhat_mat, 'UniformOutput', false');

    %back to original dimension    
    ZhatHZhat = num2cell( vertcat( ZhatHZhat{:} ), 1 );
    ZhatHZhat = cellfun( @(A)( reshape(reshape(A, [K, sy*sx]).', [sy,sx,K]) ), ...
        ZhatHZhat, 'UniformOutput', false');
    
    %|F' Sigma_k F| = block matrix with circulant matrix with absolute values only
    d_maj_freq = cellfun( @(A)( fft2(abs(ifft2(A))) ), ZhatHZhat, 'UniformOutput', false');
    
    %sum_{k'} ( P_S |F' Sigma_{k'} F| P'_S * 1vec ) for each kernel
    PSt1_freq = fft2( PSt(ones(size_kernel)) );
    Md = cellfun( @(A)( sum( PS(real(ifft2(A .* PSt1_freq))),3) ), ...
        d_maj_freq, 'UniformOutput', false');
    Md = cat(3, Md{:});
    
else   
   error('Err: Choose the appropriate  majorization matrix type.'); 
end

return;


function Mz = maj_for_z(dhat, Mz_type, size_z)
% Compute majorizer for sparse code updates

%Size
sy = size_z(1); sx = size_z(2); K = size_z(3); L = size_z(4);

%Permute spectra: each cell corresponds to frequency
dhat_mat = num2cell( reshape( dhat, sy * sx, [] ), 2 );

if strcmp( Mz_type, 'D')
    %Diagonal preconditioner in 
    %Lem. 4.5 of DOI: 10.1109/TIP.2017.2761545
    
    %construct { Sigma^{prime}_k : k=1,...,D }
    DhatHDhat = cellfun(@(A)(sum( abs(A'*A), 2 )), dhat_mat, 'UniformOutput', false');

    %reshape to sqrt(N')-by-sqrt(N')-by-D
    z_maj_freq = reshape( permute(cell2mat(DhatHDhat), [2,1]), [sy,sx,K] );
    % z_maj_freq = reshape( permute(vertcat(DhatHDhat{:}), [2,1]), [sy,sx,k] );    %slightly faster

    %|F' Sigma_k F| = circulant matrix with absolute values only
    z_maj_freq = fft2( abs( ifft2(z_maj_freq) ) );
    
    %|F' Sigma_k F| * 1vec
    Mz = real( ifft2( z_maj_freq .* fft2( ones([sy sx K]) ) ) );
    Mz = repmat( Mz, [1 1 1 L] );
    
elseif strcmp( Mz_type, 'Dtgt')
    %Diagonal preconditioner in
    %Lem. 4.7 of DOI: 10.1109/TIP.2017.2761545
    %| Note: Compared to 'D1', 'D2' obtains sharper majorization
    %|       (i.e., faster convgergence), but requires more compt.
    
    %reshpae to sqrt(N')-by-sqrt(N') freq-cell:
    %ith columns of cells correspond to rows of [ Lambda' Lambda ]_{i,*}
    DhatHDhat = cellfun(@(A)( ( A'*A ).' ), dhat_mat, 'UniformOutput', false');

    %back to original dimension    
    DhatHDhat = num2cell( vertcat( DhatHDhat{:} ), 1 );
    DhatHDhat = cellfun( @(A)( reshape(reshape(A, [k, sy*sx]).', [sy,sx,K]) ), ...
        DhatHDhat, 'UniformOutput', false');
    
    %|F' Sigma_k F| = block matrix with circulant matrix with absolute values only
    z_maj_freq = cellfun( @(A)( fft2(abs(ifft2(A))) ), ...
        DhatHDhat, 'UniformOutput', false');
    
    %sum_{k'} ( |F' Sigma_k' F| * 1vec ) for each kernel
    one_freq = fft2( ones([sy sx k]) );
    Mz = cellfun( @(A)( sum(real(ifft2(A .* one_freq)),3) ), ...
        z_maj_freq,  'UniformOutput', false');
    Mz = cat(3, Mz{:});
    Mz = repmat( Mz, [1 1 1 L] );

else  
    error('Err: Choose the appropriate majorization matrix option.'); 
end

return;





%%%%%%%%%%%%%%%%%%%% MISC %%%%%%%%%%%%%%%%%%%%

function u = KernelConstraintProj( u, Md, Md_type )
        
u_norm = sum(sum(u.^2, 1),2);
kern_idx = u_norm >= 1;

if sum(kern_idx(:)) ~= 0

    if strcmp( Md_type, 'I')
        %Projection to unit sphere constraint
        
        u_norm = repmat( u_norm, [size(u,1), size(u,2), 1] );
        u( u_norm >= 1 ) = u( u_norm >= 1 ) ./ sqrt(u_norm( u_norm >= 1 ));
            
    elseif strcmp( Md_type, 'D') || strcmp( Md_type, 'Dtgt')
        %Solve (1/2) u' Md u - y' Md u, s.t. u'u <= 1
        
        u_targ = u(:,:,kern_idx);
        Md_targ = Md(:,:,kern_idx);
        
        Md_cell = num2cell( Md_targ, [1,2] );
        Mdu2_cell = num2cell( ( Md_targ .* u_targ ).^2, [1, 2] );

        %Parameters for Newton's method
        rad = 1;      %unit sphere
        lambda0 = 0;    %initial value
        tol = 1e-6;     %tolerance
        max_iter = 10;  %max number of iterations

        %Optimization with Newton's method
        lambda_opt = cell2mat( cellfun( @(A,B) Newton( @(x) f_lambda(A, B, x), @(x) df_lambda(A, B, x), rad, lambda0, tol, max_iter ), ...
            Mdu2_cell, Md_cell, 'UniformOutput', false' ) );

        u_targ = ( Md_targ ./ ( Md_targ + repmat(lambda_opt, [size(u_targ,1), size(u_targ,2), 1] ) ) ) .*  u_targ;

        u(:,:,kern_idx) = u_targ;
        
    else
        error('Err: Choose the appropriate majorization matrix option.');
    end
    
end

return;


function fval = f_lambda(Mdu2, Md, lambda)

    fval = sum( Mdu2(:) ./ (Md(:) + lambda).^2 );

return;


function dfval = df_lambda(Mdu2, Md, lambda)

    dfval = -2 * sum( Mdu2(:) ./ (Md(:) + lambda).^3 );

return;





%%%%%%%%%%%%%%%%%%%% MISC %%%%%%%%%%%%%%%%%%%%

function f_val = objectiveFunction( z, d_hat, x, alpha, psf_radius, center, psf_s, size_z, size_xpad)

    sy = size_z(1); sx = size_z(2); K = size_z(3); L = size_z(4);   %size

    %Sparse representation error + sparsity
    zhat = reshape( fft2(reshape(z,size_z(1),size_z(2),[])), size_z );
    Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,L]) .* zhat, 3), size_xpad) ));
    
    if mod(psf_s,2) == 0
        f_z = 0.5 * norm( reshape( Dz(center(1):end - (psf_s-center(1)), ...
            center(2):end - (psf_s-center(2)), :) - x, [], 1) , 2 )^2;
    else        
        f_z = 0.5 * norm( reshape( Dz(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:) - x, [], 1) , 2 )^2;
    end
    g_z = alpha * sum( abs( z(:) ), 1 );
    
    %Function val
    f_val = f_z + g_z;
    
return;


function [] = display_func(iterate_fig, filter_fig, d, d_hat, z_hat, x, size_xpad, size_z, psf_radius, iter)

    sy = size_z(1); sx = size_z(2); k = size_z(3); L = size_z(4);   %size

    figure(iterate_fig);
    Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,L]) .* z_hat, 3), size_xpad) ));
    Dz = Dz(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:);

    subplot(3,2,1), imagesc(x(:,:,2));  axis image, colormap gray; colorbar; title('Orig');
    subplot(3,2,2), imagesc(Dz(:,:,2)); axis image, colormap gray; colorbar; title(sprintf('Local iterate %d',iter));
    subplot(3,2,3), imagesc(x(:,:,4));  axis image, colormap gray; colorbar;
    subplot(3,2,4), imagesc(Dz(:,:,4)); axis image, colormap gray; colorbar;
    subplot(3,2,5), imagesc(x(:,:,6));  axis image, colormap gray; colorbar;
    subplot(3,2,6), imagesc(Dz(:,:,6)); axis image, colormap gray; colorbar;
    drawnow;

    figure(filter_fig);
    sqr_k = ceil(sqrt(size(d,3)));
    pd = 1;
    d_disp = zeros( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
    for j = 0:size(d,3) - 1
        d_curr = d(:,:,j+1);
        d_disp( floor(j/sqr_k) * (size(d_curr,1) + pd) + pd + (1:size(d_curr,1)) , mod(j,sqr_k) * (size(d_curr,2) + pd) + pd + (1:size(d_curr,2)) ) = d_curr;
    end
    imagesc(d_disp), colormap gray, axis image, colorbar, title(sprintf('Local filter iterate %d',iter));
    drawnow;
        
return;