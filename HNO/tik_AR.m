function f = tik1d_transfAR(g, PSF, center, lambda);
%  
%       f = tik1d_transfAR(g, PSF, center, lambda);
%
%  Compute restoration using a Tikhonov filter with AR BCs. 
%
%  Input:
%        g  -  Array containing blurred signal.
%      PSF  -  Array containing the point spread function.
%   center  -  Centeral index of the PSF.
%   lambda  -  Regularization parameter.
%
%  Output:
%        f  -  Array containing computed restoration.

%  M. Donatelli 10-06-08

% Check ord
if (nargin < 5)
   ord = 0;
end

n = length(g);
d = eigAR(PSF,center,n);
phi = d ./ (d.^2 + lambda^2);
ghat = idARt(g);
fhat = phi .* ghat;
f = real(dARt(fhat));
