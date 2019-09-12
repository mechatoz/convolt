function y = dst(x)
%  
%       y = dst(x);
%
%  Compute the discrete sine transform of x
%  
%  Input: x  -  vector to transform (if x is a matrix each column is 
%               tranformed)  
%
%  Output:
%         y  -  discrete sine transform of x 
%
%  See also: idst.
%
%  Reference: "Computational Frameworks for the Fast Fourier Transform"
%             by C. Van Loan, SIAM, 1992.

% M. Donatelli 05/20/2006

[n,m] = size(x);
xtr = 0;
if (n==1)
    xtr = 1;
    x = x(:);
    [n,m] = size(x);    
end

z = zeros(1,m);
xt = [z; x; z; -flipud(x)];
yt = fft(xt);
y = sqrt(-1)/2 * yt(2:n+1,:);

if isreal(x), y=real(y); end
if (xtr==1), y=y.'; end
