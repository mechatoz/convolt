function X = tik_AR2(B, PSF, center, alpha)
%TIK_AR2 Tikhonov image deblurring using the AR algorithm.
%
%function X = tik_AR2(B, PSF, center, alpha)
%
%            X = tik_AR2(B, PSF, center);
%            X = tik_AR2(B, PSF, center, alpha);
%
%  Compute restoration using a AR-based Tikhonov filter, 
%  with the identity matrix as the regularization operator.
%
%  Input:
%        B  Array containing blurred image.
%      PSF  Array containing the point spread function; same size as B.
%   center  [row, col] = indices of center of PSF.
%    alpha  Regularization parameter.
%
%  Output:
%        X  Array containing computed restoration.

% Last revised April 14, 2008.
% Author: Marco Donatelli.

%
% Check number of inputs and set default parameters.
%
if (nargin < 3)
   error('B, PSF, and center must be given.')
end
if (nargin < 4)
   alpha = [];
end

%
% Use the eigAR2 to compute the eigenvalues of the symmetric 
% antireflective blurring matrix.
%
bhat = idARt2(B);
bhat = bhat(:);
S = eigAR2(PSF,center);
s = S(:);
if (ischar(alpha) | isempty(alpha))
    alpha = gcv_tik(s, bhat);
end 

%
% Compute the Tikhonov regularized solution.
%
D = s.^2 + abs(alpha)^2;
bhat = s .* bhat;
xhat = bhat ./ D;
xhat = reshape(xhat, size(B));
X = dARt2(xhat);
