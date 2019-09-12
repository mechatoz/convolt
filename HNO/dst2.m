function y = dst2(x)
%
%      y = dst2(x);
%
%  Compute the 2D discrete sine transform of x
%  
%  Input: x  -  2D array to transform  
%
%  Output:
%         y  -  2D discrete sine transform of x 
%
%  See also: dst, idst2.

% M. Donatelli 05/20/2006

[n,m] = size(x);
if min(n,m) == 1
    y = dst(x);
else
    y = dst(dst(x).').';
end