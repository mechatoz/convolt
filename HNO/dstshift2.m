function Ps = dstshift2(PSF, center)
%
%       Ps = dstshift2(PSF, center)
%
%  Create an array containing the first column of a tau 2D matrix
%
%  Input:
%      PSF - Array containing the point spread function.
%   center - [row, col] = indices of center of the PSF.
%
%  Output:
%       Ps - Array containing first column of blurring matrix.

% Author: Marco Donatelli
% Date: 05/16/2008

[m,n] = size(PSF);

if nargin == 1
  error('The center must be given.')
end

i = center(1);
j = center(2);
k = min([i-1,m-i,j-1,n-j]);

%
% The PSF gives the entries of a central column of the blurring matrix.
% The first column is obtained by reordering the entries of the PSF; 
%
PP = PSF(i-k:i+k,j-k:j+k);

Z1 = diag(ones(k+1,1),k);
Z2 = diag(ones(k-1,1),k+2);

PP = Z1*PP*Z1' - Z1*PP*Z2' - Z2*PP*Z1' + Z2*PP*Z2';

Ps = zeros(m,n);
Ps(1:2*k+1,1:2*k+1) = PP;