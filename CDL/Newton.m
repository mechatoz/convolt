function [ x, ex ] = Newton( f, df, alpha, x0, tol, nmax )

%| Newton:   
%| Accelerated Newton's method for finding zeroes of a real-valued func.
%|
%| [Input]
%| f: funtion
%| df: derivative of function
%| x0: initial approx.
%| tol: tolerance
%| nmax: max number of iterations
%|
%| [Output]
%| x: aproximation to root
%| ex: difference b/w current and previous updates
%|
%| Copyright 2019-07-31, Il Yong Chun, University of Michigan

if nargin == 4
    tol = 1e-4;
    nmax = 1e1;
elseif nargin == 5
    nmax = 1e1;
elseif nargin ~= 6
    error('newton: invalid input parameters');
end

xold = x0;

%Accelerated Newton method
fx = f(x0);
x = x0 - 2 * ( fx / df(x0) ) * ( sqrt(fx) / alpha - 1 );

%Classical Newton method
%x = x0 - (f(x0) - alpha^2) / df(x0);

ex = abs(x-x0);

k = 2;
while ( k <= nmax ) && ( x > xold ) && ( ex >= tol )
    xold = x;

    %Accelerated Newton method
    fx = f(x);
    x = x - 2 * ( fx / df(x) ) * ( sqrt(fx) / alpha - 1 );

    %Classical Newton method
    %x = x - (f(x) - alpha^2) / df(x);

    ex = abs(x-xold);
end

end