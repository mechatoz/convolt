function d = eigAR(PSF,center,n)
%  
%       d = eigAR(PSF,center,n)
%
%  Compute eigenvalues of the Anti-Reflective matrix.
%
%  Reference: A. Aric\`{o}, M. Donatelli and S. Serra-Capizzano "Spectral 
%   analysis of the anti-reflective algebra", Linear Algebra Appl., 428 
%   (2008), pp. 657--675 .
%
%  Input:
%      PSF  -  Array containing the point spread function.
%   center  -  Index of center of the PSF.
%
%  Optional Intputs:
%        n  -  Partial size of the AR matrix (n > length(PSF)/2)
%              (default: n = length(PSF)) 
%
%  Output:
%        d  -  Eigenvalues of the AR matrix.

%  M. Donatelli 10/28/2006


% check input n
if nargin < 3 
    n = length(PSF);
end
m = n-2;
if m < ceil(length(PSF))/2
    error('Must be n > length(PSF)/2+2')
end

% eigenvalues
e1 = zeros(m,1);  e1(1) = 1;
dt = dst(dstshift(PSF, center, m)) ./ dst(e1);
d = [sum(PSF); dt; sum(PSF)];
