function X = idARt(Y)
%IDART  Inverse Discrete Anti-Reflective Transform.
%
%       X = idARt(Y) returns the inverse "discrete Anti-Reflective transform" of Y.
%       The vector X is the same size as Y and contains the inverse discrete 
%       anti-reflective transform coefficients.
%
%       If Y is a matrix, the idARt operation is applied to each
%       column. This transform can be inverted using Y=dARt(X).
%
%       Y --> idARt(Y) is defined by:
%
%             |   1         |
%       Y --> | -Q*p Q -Q*p'| * Y
%             |          1  |
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
     error('Wrong number of input arguments: should be 1')
end

n = size(Y,1);
if n<3
     error('size(Y,1) should be >2.')
end

% not used (useful x the scaled version)
% alpha = sqrt( n*(2*n-1) / (6*n-6) );

if n==3
    % inner n-2 part is #=1 and dst should **not** be applied
    X  = [ 
        Y(1,:)
        Y(2,:) - (Y(1,:)+Y(3,:))/2
        Y(3,:)
    ];
else
    % X = dARt(Y)
    Q_p  = cot( pi*[1:n-2].'/(2*n-2) ) / sqrt(2*n-2);
    Q_p1 = (2* mod([1:n-2].',2) -1) .* Q_p; % x->flipud(x) is "tau(-(n+1)*x)"
    X  = [ 
        Y(1,:)
        -Q_p*Y(1,:) + sqrt(2/(n-1)) *dst(Y([2:n-1],:)) - Q_p1*Y(n,:)% scaled dst
        Y(n,:)
    ];
end