% This script propagates a Gaussian wave packet in one dimension.
% The evolution is known analytically; this implementation serves 
% to test four different implementations of the kinetic energy 
% operator:
% 1 - Finite difference, three point rule
% 2 - Finite difference, five point rule
% 3 - Pseudo spectral, FFT (fast Fourier transform)
% Case 3 leads to periodic boundary conditions, the other ones 
% fulfil Dirichlet boundary conditions.
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
% N - number of grid points, should be 2^n
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
% The FFT propagation scheme calls fft and ifft.

% Clear memory and set format for printouts
clear all
format short e

% Physical parameters:
x0 = -20;
k0 = 3;
sigmaK = .2;
t0 = 5;

% Numerical time parameters:
Ttotal = 35;
dt = 0.05;

% Grid parameters
L = 100;
N = 256;              % For FFT's sake, we should have N=2^n


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Set up the grid.

% Finite difference and FFT-version:
x = linspace(-.5,.5,N)'*L;
h = x(2)-x(1);                                          % Spatial step size
wavenumFFT = 2*(pi/L)*[(0:N/2-1), (-N/2:-1)]';          % Momentum vector, FFT
Hfft = -1/2*ifft(diag((1i*wavenumFFT).^2)*fft(eye(N)));
%
% Construct finite difference Hamiltonians 
% (just kinetic energy) and propagators
%

% Three point finite difference, 
%f''(x) = ( f(x-h)-2f(x)+f(x+h) ) / 2 h^2
e=ones(N,1); 
Hfd3=spdiags([e -2*e e],-1:1,N,N);                  % Tri-diagonal matrix
Hfd3=-1/2*Hfd3/h^2;

% Five point finite difference, 
% f''(x) = ( -f(x-2h)+16f(x-h)-30 f(x) + 16 f(x+h) - f(x+2h) ) / 12 h^2
Hfd5=spdiags([-e 16*e -30*e 16*e -e],-2:2,N,N);     % Band-diagonal matrix
Hfd5=-1/2*Hfd5/(12*h^2);

% Construct propagators for finite difference schemes (matrix exponentials)
Ufd3 = expm(-1i*Hfd3*dt);
Ufd5 = expm(-1i*Hfd5*dt);
Ufft = expm(-1i*Hfft*dt);

clear Hfd3 Hfd5 Hfft;                               % Remove obsolete stuff from memory

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
PsiFD5=psi0Gauss;
PsiFFT=psi0Gauss;
PsiAnalyticalSq=AnalyticalGaussWFsq(x,x0,0,sigmaK,t0,k0);
t=0;
n_dt = floor(Ttotal/dt)+1;          % Number of time steps

% Create plots
figure(1)
pl1 = plot(x,PsiAnalyticalSq,'k-');
hold on
pl2 = plot(x,abs(PsiFD3).^2,'b:');
pl3 = plot(x,abs(PsiFD5).^2,'r--');
pl4 = plot(x,abs(PsiFFT).^2,'m-.');
hold off
% Find maximum value of |\Psi(x)|^2 (for t=t0)
MaxValX=max(AnalyticalGaussWFsq(x,x0,t0,sigmaK,t0,k0));
% Set axis
axis([min(x) max(x) 0 1.1*MaxValX])
xlabel('x')
legend('Analytical','FD3','FD5','FFT')

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

  % Finite difference schemes
  PsiFD3=Ufd3*PsiFD3;
  PsiFD5=Ufd5*PsiFD5;
  PsiFFT=Ufft*PsiFFT;
  
  % Analytical wave function
  PsiAnalyticalSq=AnalyticalGaussWFsq(x,x0,t,sigmaK,t0,k0);
  
  % Plot wave functions on the fly
  set(pl1, 'ydata', PsiAnalyticalSq);  
  hold on
  set(pl2, 'ydata', abs(PsiFD3).^2);
  set(pl3, 'ydata', abs(PsiFD5).^2);
  set(pl4, 'ydata', abs(PsiFFT).^2);
  hold off
  drawnow
  % Set axis
  axis([min(x) max(x) 0 1.1*MaxValX])
end