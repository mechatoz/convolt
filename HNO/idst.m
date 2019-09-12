function y = idst(x)
%  
%       y = Idst(x);
%
%  Compute the inverse discrete sine transform of x
%  
%  Input: x  -  vector to transform (if x is a matrix each column is 
%               tranformed)  
%
%  Output:
%         y  -  inverse discrete sine transform of x 
%
%  See also: dst.

% M. Donatelli 05/20/2006

n = length(x);
y = 2/(n+1) * dst(x);

