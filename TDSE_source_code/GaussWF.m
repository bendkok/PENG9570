function Psi=GaussWF(x,x0,SigmaP,tau,p0)

% This function is an initial wave packet which will undergo
% an evolution in which its absolute square remains Gaussian.
% The input variables/parameters are
% x         - the position
% x0        - mean position
% SigmaP    - the momentum width
% tau       - the time at which the wave packet is at its narrowest
% p0        - the mean momentum of the wave packet
% 
% x may be an array, while all other parameters must be scalars.

% Normalization factor
N=sqrt(sqrt(2)*SigmaP/(sqrt(pi)*(1-2i*SigmaP^2*tau)));
% The full wave function
Psi=N*exp(-SigmaP^2*(x-x0).^2/(1-2i*SigmaP^2*tau)).*exp(1i*p0*x);