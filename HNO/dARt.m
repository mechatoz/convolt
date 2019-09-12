function Y = dARt(X)
%DART   Discrete Anti-Reflective Transform.
%
%       Y = dARt(X) returns the "discrete Anti-Reflective transform" of X.
%       The vector Y is the same size as X and contains the discrete 
%       anti-reflective transform coefficients.
%
%       If X is a matrix, the dARt operation is applied to each
%       column. This transform can be inverted using X=idARt(Y).
%
%       X --> dARt(X) is defined by:
%
%             | 1     |
%       X --> | p Q p'| * X
%             |     1 |
%
%       where:
%           - p  = [n-2:-1:1].' /(n-1),
%           - p' = [1:n-2].' /(n-1),                             _______
%           - Q  = DST-I(n-2) (which is unitary i.e. scaled by \/2/(n-1) )
%
%       *** the condition size(X,1) >= 3 is required ***
%
%  Reference: A. Aric\`{o}, M. Donatelli, J. Nagy, and S. Serra-Capizzano 
%       "The Anti-Reflective Transform and Regularization by Filtering"

%
% A. Arico' 10/20/2006

if nargin ~= 1
     error('Wrong number of input arguments: should be 1.')
end

n = size(X,1);
if n<3
     error('size(X,1) should be >2.')
end

% not used (useful 4 the scaled version)
% alpha = sqrt( n*(2*n-1) / (6*n-6) );

if n==3
    % inner n-2 part is #=1 and dst should **not** be applied
    Y  = [ 
        X(1,:)
        X(2,:) + (X(1,:)+X(3,:))/2
        X(3,:)
    ];
else
    % Y = dARt(X)
    p1 = [1:n-2].' /(n-1);
    p  = flipud(p1);
    Y  = [ 
        X(1,:)
        p*X(1,:) + sqrt(2/(n-1))* dst(X([2:n-1],:)) + p1*X(n,:) % dst is scaled
        X(n,:)
    ];
end