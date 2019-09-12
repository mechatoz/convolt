% AR package. 
% Version 1.0  24-July-08. 
% Copyright (c) 2008 by A. Arico' and M. Donatelli.
%
% Matlab functions for 1D and 2D anti-reflective (AR) boundary conditions 
% (BC). Eigenvalues computation, AR transform and Tikhonov regularization.
%
% Requires Matlab version 6.5 or later versions.
%
% The package uses functions from the HNO package of Hansen, Nagy and 
% O'leary available at www.siam.org/books/fa03 and based on the book
%   Deblurring Images - Matrices, Spectra, and Filtering
%   P. C. Hansen, J. G. Nagy, and D. P. O'Leary
%   SIAM, Philidelphia, 2006.
% thus the above package should be uploaded and be accessible to MATLAB.
%    
% This package is based on the following papers:
% - A. Arico', M. Donatelli, and S. Serra-Capizzano, "Spectral analysis 
%   of the anti-reflective algebra", Linear Algebra Appl., 428 (2008), 
%   pp. 657--675 .
% - A. Arico', M. Donatelli, J. Nagy, and S. Serra Capizzano, 
%   "The Anti-Reflective Transform and Regularization by Filtering",
%    Numerical Linear Algebra in Signals, Systems, and Control., in Lecture 
%   Notes in Electrical Engineering edited by S. Bhattacharyya, R. Chan, 
%   V. Olshevsky, A. Routray, and P. Van Dooren, Springer Verlag, 2008.
%  
% Demonstration. 
%   example - Demonstrates how to do 2D Tikhonov regularization with
%             AR BCs and it gives a comparisson with reflective BCs.
%  
% 1D functions. 
%   dst      - discrete sine transform
%   idst     - inverse discrete sine transform
%   dstshift - first column of the tau matrix
%   eigAR    - eigenvalues of the AR matrix
%   dARt     - discrete AR transform
%   idARt    - inverse discrete AR transform
%   tik_AR   - Tikhonov regularization
%    
% 2D functions. 
%   dst2      - discrete sine transform
%   idst2     - inverse discrete sine transform
%   dstshift2 - first column of the tau matrix
%   eigAR2    - eigenvalues of the AR matrix
%   dARt2     - discrete AR transform
%   idARt2    - inverse discrete AR transform
%   tik_AR2   - Tikhonov regularization
