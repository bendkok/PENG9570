function V=SmoothRectangularPot(x,V0,w,s)

% This function is a smooth version of a rectangular potential 
%to be used in simuations of wave packet simulations.
% % The input variables/parameters are
% x         - the position
% V0        - the hight of the potential; it could be negative
% w         - the width of the potential
% s         - the degree of "smoothness"

V=V0./(exp(s*(abs(x)-w/2))+1);