function y = idst2(x)
%
%      y = idst2(x);
%
%  Compute the 2D inverse discrete sine transform of x
%  
%  Input: x  -  2D array to transform  
%
%  Output:
%         y  -  2D inverse discrete sine transform of x 
%
%  See also: idst, dst2.

% M. Donatelli 05/20/2006

[n,m] = size(x);
if min(n,m) == 1
    y = idst(x);
else
    y = idst(idst(x).').';
end
