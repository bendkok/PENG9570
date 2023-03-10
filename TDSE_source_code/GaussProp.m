% This script propagates a Gaussian wave packet in one dimension.
% The evolution is known analytically; this implementation serves 
% to test finite different implementations of the kinetic energy 
% operator:
%
%
% Physical input parameters:
%
% x0 - initial mean position
% k0 - initial mean velocity of the wave packet
% t0 - the time at which the wave packet is at its narrowest, spatially
% sigmaK - the momentum width of the wave packet
%
%
% Numerical parameters:
% 
% Ttotal - the duration of the propagation
% dt - numerical time step
% N - number of grid points
% L - the size of the numerical domain; it extends from -L/2 to L/2
% 
% All input parameters are hard coded initially.
%
% 
% Function calls
% 
% In order to initiate the wave functions, the function file GaussWF is 
% called. The analytical solution for |\Psi(x;t)|^2, which is plotted on 
% the fly along with the numerical extimates, is provided in the function
% file AnalyticalGaussWFsq.

% Clear memory and set format for printouts
clear all
format short e

% Physical parameters:
x0 = -20;
k0 = 3;
sigmaK = .2;
t0 = 5;

% Numerical time parameters:
Ttotal = 10;    % Total duration
dt = 0.025;      % Numerical time step

% Grid parameters
L = 100;        % Extension (from -L/2 to L/2)
N = 150;        % Spatial grid points          


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Set up the grid.

% Finite difference and FFT-version:
x = linspace(-.5,.5,N)'*L;
h = x(2)-x(1);                                          % Spatial step size

% Three point finite difference, 
%f''(x) = ( f(x-h)-2f(x)+f(x+h) ) / 2 h^2
e=ones(N,1); 
Hfd3=spdiags([e -2*e e],-1:1,N,N);                  % Tri-diagonal matrix
Hfd3=-1/2*Hfd3/h^2;
% Needless to say: We could also have constructed Hfd3 by using a for-loop.

% Construct propagators for finite difference schemes (Crank-Nicolson)
Ufd3 = inv(eye(N)+1i*Hfd3*dt/2)*(eye(N)-1i*Hfd3*dt/2);

clear Hfd3;                               % Remove obsolete stuff from memory

%
% Initial Gauss packet
%

% Write parameters to screen
sigmaX0=1/sigmaK*sqrt(1+sigmaK^4*t0^2);             % Spatial width
sigmaXmin=1/sigmaK;
disp(['Initial spatial width: ',num2str(sigmaX0),...
', minimal widht: ',num2str(sigmaXmin)])            
meanE=.5*(k0^2+sigmaK^2/2);                         % Energy
DeltaE=.25*sigmaK^2*(2*k0^2+sigmaK^2/2);            % Energy width
disp(['Mean energy: ',num2str(meanE),', width: ',num2str(DeltaE)])

% Gaussian wave packet with minimal width at t=t0, 
% initial 'position' x0 and mean momentum k0.
psi0Gauss=GaussWF(x,x0,sigmaK,t0,k0);

% Initiate all wave functions (including the analytical one) and time
PsiFD3=psi0Gauss;
PsiAnalyticalSq=AnalyticalGaussWFsq(x,x0,0,sigmaK,t0,k0);

t=0;
n_dt = floor(Ttotal/dt)+1;          % Number of time steps

% Create plots
figure(1)
pl1 = plot(x,PsiAnalyticalSq,'k-');
hold on
pl2 = plot(x,abs(PsiFD3).^2,'b:');
hold off
% Find maximum value of |\Psi(x)|^2 (for t=t0)
MaxValX=max(AnalyticalGaussWFsq(x,x0,t0,sigmaK,t0,k0));
% Set axis
axis([min(x) max(x) 0 1.1*MaxValX])
xlabel('x')
legend('Analytical','FD3')

%
% Propagate
%
ProgressOld=0;

for k = 1:n_dt
  % Write progress to screen
  ProgressNew=floor(k/n_dt*10);
  if ProgressNew~=ProgressOld
    disp(['Progress: ',num2str(10*ProgressNew),'%'])
    ProgressOld=ProgressNew;
  end

  % Update time
  t=t+dt;

  % Take a time step with finite difference scheme
  PsiFD3=Ufd3*PsiFD3;
  
  % Analytical wave function
  PsiAnalyticalSq=AnalyticalGaussWFsq(x,x0,t,sigmaK,t0,k0);
  
  % Plot wave functions on the fly
  set(pl1, 'ydata', PsiAnalyticalSq);  
  hold on
  set(pl2, 'ydata', abs(PsiFD3).^2);
  hold off
  drawnow
  % Set axis
  axis([min(x) max(x) 0 1.1*MaxValX])
end