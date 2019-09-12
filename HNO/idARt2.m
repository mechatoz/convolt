function y = idARt2(x)
%
%      y = idARt2(x);
%
%  Compute the 2D inverse discrete antireflective transform of x
%  
%  Input: x  -  2D array to transform  
%
%  Output:
%         y  -  2D inverse discrete antireflective transform of x 
%
%  See also: idARt, dARt2.

% M. Donatelli 18/04/08

[n,m] = size(x);
if min(n,m) == 1
    y = idARt(x);
else
    y = idARt(idARt(x).').';
end
