function PsiSq=AnalyticalGaussWFsq(x,x0,t,SigmaP,tau,p0)

% This function is an the absolute square of a travelling Gaussian wave packet.
% The input variables/parameters are
% x         - the position
% x0        - mean position
% t         - time
% SigmaP    - the momentum width
% tau       - the time at which the wave packet is at its narrowest
% p0        - the mean momentum of the wave packet
% 
% x may be an array, while all other parameters must be scalars.

% Normalization factor
Nsq=sqrt(2/pi)*SigmaP/sqrt(1+4*SigmaP^4*(t-tau)^2);
% The full wave function
PsiSq=Nsq*exp(-2*SigmaP^2*(x-x0-p0*t).^2/(1+4*SigmaP^4*(t-tau)^2));