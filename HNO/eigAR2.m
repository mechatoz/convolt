function d = eigAR2(PSF,center)
%
%       d = eigAR2(PSF,center)
%
%  Compute eigenvalues of the 2D Anti-Reflective matrix.
%
%  IMPORTANT: it assumes that the last two rows and columns of the PSF
%             are negligible!
%
%  Reference: A. Aric\`{o}, M. Donatelli and S. Serra-Capizzano "Spectral 
%             analysis of the anti-reflective algebras and applications", 
%             Linear Algebra Appl., 428 (2008), pp. 657-675.
%
%  Input:
%      PSF  -  Array containing the point spread function.
%   center  -  Index of center of the PSF.
%
%  Optional Intputs:
%        n  -  Partial size of the AR matrix (n > max(size(PSF))/2)
%              (default: n = max(size(PSF))) 
%
%  Output:
%        d  -  Eigenvalues of the 2D AR matrix.
% 
%  Reference: "Spectral analysis of the anti-reflective algebra"
%             by A. Arico', M. Donatelli, and S Serra-Capizzano,
%             Linear Algebra Appl., 428 (2008), pp. 657--675.

% 
%  M. Donatelli 05/16/2008

[n1,n2] = size(PSF);

% eigenvalues at the edges are 1d AR problems (see the reference)
P1 = sum(PSF);
d(1,:) = eigAR(P1,center(2),n2);
d(n1,:) = d(1,:);
P2 = sum(PSF');
d(:,1) = eigAR(P2,center(1),n1);
d(:,n2) = d(:,1);

% inner dst2 part
e1 = zeros(size(PSF)-[2,2]); e1(1,1) = 1;
%dt = dst2(dstshift2(PSF(2:n1-1,2:n2-1), center-[1,1])) ./ dst2(e1);
dt = dst2(dstshift2(PSF(1:n1-2,1:n2-2), center)) ./ dst2(e1);
d(2:n1-1,2:n2-1)= dt;

