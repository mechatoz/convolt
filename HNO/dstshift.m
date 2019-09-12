function c = dstshift(PSF, center, n);
%  
%       c = dstshift(PSF, center, n);
%
%  Create an array containing the first column of a tau matrix 
%  (diagonalizable by DSTs).
%
%  Reference: D. Bini and M. Capovani, "Spectral and computational 
%             properties of band symmetric Toeplitz matrices" Linear Algebra 
%             Appl., 52/53, pp. 99-125, 1983.
%
%  Input:
%      PSF  -  Array containing the point spread function.
%   center  -  Centeral index of the PSF.
%
%  Optional Intputs:
%        n  -  Partial size of the tau matrix (n > length(PSF)/2)
%              (default: n = length(PSF)) 
%
%  Output:
%        c  -  First column of the tau matrix.

%  M. Donatelli 10/22/2006

% check input n
if nargin < 3 
    n = length(PSF);
end
if n < ceil(length(PSF)/2)
    error('Must be n > length(PSF)/2')
end

% first column
c = PSF(center:end);
c(1:end-2) = c(1:end-2) - c(3:end);
lc = length(c);
c(lc+1:n)=zeros(1,n-lc);
c = c(:);