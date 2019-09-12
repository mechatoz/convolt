function y = dARt2(x)
%
%      y = dARt2(x);
%
%  Compute the 2D discrete antireflective transform of x
%  
%  Input: x  -  2D array to transform  
%
%  Output:
%         y  -  2D discrete antireflective transform of x 
%
%  See also: dARt, idARt2.

% M. Donatelli 18/04/08

[n,m] = size(x);
if min(n,m) == 1
    y = dARt(x);
else
    y = dARt(dARt(x).').';
end