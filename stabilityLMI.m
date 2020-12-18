%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diffusion-reaction equation coupled with an ODE
% z_t(t,th) = delta z_thth(t,th) + lambda z(t,th)
% x_t(t) = A x(t) + B z^o(t) 
% z^i(t) = C x(t) + D z^o(t)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; 

%% Parameters
% PDE
a = 0; b = 1;
len = b-a;
sys.PDE.delta = 1;
sys.PDE.lambda = 0.5;
% ODE
sys.ODE.A = -1;
sys.ODE.B = [1 0];
sys.ODE.C = [1; 0];
sys.ODE.D = zeros(2);
sys.ODE.ss = ss(sys.ODE.A,sys.ODE.B,sys.ODE.C,sys.ODE.D,'OutputName',{'y1','y2'});

%% Sufficient stability condition at order ntest
reslmi = 0;
for ntest = 1:4
    % Model at order ntest
    sys.ntest.ss = sysn(sys.ODE.ss,sys.PDE.delta, sys.PDE.lambda, len, ntest); % Creation of PDE ss model at order ntest
    % LMI test
    res = lmin(sys.ntest.ss,sys.PDE.delta, sys.PDE.lambda, len, ntest); % Creation of PDE ss model at order nsim
    if res == 1
        reslmi = ntest;
        break
    end
end

if reslmi == 0
    disp('LMI is unfeasible')
else
    disp(['LMI ensures stability from order ' int2str(reslmi)])
end

function [out] = sysn(sys, delta, lambda, len, n)
% Approximation of diffusion-reaction equation at order n
% z_t(t,th) = delta z_thth(t,th) + lambda z(t,th)
% z^i(t) = z_th(t,1/0);
% z^o(t) = z(t,1/0)
% Input: delta, lambda, n
% Output:  state space model at order n
    A = sys.A; B = sys.B; C = sys.C; D = sys.D;
    nlist = (0:n)';
    I = 1.^nlist; % Boundary th=1
    Ib = (-1).^nlist; % Boundary th=0
    In = 1/len*diag(2*nlist+1); % Norm
    Ln = tril(I*I'-Ib*Ib',-1); % Derivation
    Co = [Ib'*In;I'*In]; % Output z(a/b)
    Ci = [Ib'*In*triu(I*I'-Ib*Ib',1)*In;I'*In*triu(I*I'-Ib*Ib',1)*In]; % Output z_th(a/b)
    Bi = [-Ib I]; % Input z_th(a/b)
    Bo = [Ln*In*Ib -Ln*In*I]; % Input z(a/b)
    Dn = 1/2*[-(-1)^n (-1)^n; 1 1];
    Cn = -Ci + D*Co;
    Bn = delta*(Bo + Bi*D);
    An = delta*(Ln*In)^2+lambda*eye(n+1)+Bn*Co;
    At = [An delta*Bi*C; B*Co A];
    Bt = [Bn; B]*Dn;
    Ct =  2*Dn'*[Cn C];
    Dt = 2*Dn'*D*Dn;
    out = ss(At,Bt,Ct,Dt,'InputName',{'wm','wp'},'OutputName',{'wdm','wdp'});
end

function res = lmin(sys, delta, lambda, len, n)
% LMI to ensure the stability of the interconnection ODE/difusion-reaction
    %% Matrices and constants
    A = sys.A;
    B = sys.B;
    C = sys.C;
    D = sys.D;
    m = size(A,1);
    k = max(0,lambda/delta*(len/(pi)^2));
    Psi = [gamman(n,k,len) 0; 0 gamman(n+1,k,len)];
    %% Lyapunov analysis
    setlmis([]) 
    P = lmivar(1,[m 1]);  % symmetric 
    % Positivity of V
    lmi0=newlmi;
    lmiterm([lmi0 1 1 P],-1,1);
    % Positivity of \dot{V}
    lmi1=newlmi;
    lmiterm([lmi1 1 1 P],1,A,'s');
    lmiterm([lmi1 1 2 P],1,B);
    lmiterm([lmi1 1 2 0],C');
    lmiterm([lmi1 2 2 0],-Psi);
    lmiterm([lmi1 2 2 1],1,D,'s');
    %% Calculations
    LMIs = getlmis;    
    options = [1e-5,0,0,0,1];
    [tmin, xfeas] = feasp(LMIs,options);
    %% Result
    if tmin < -1e-5
       res = 1;
    else
       res = 0;
    end
end

function out = gamman(n,k,len)
% Calcul de gamman
    out = ((1-k)*2*(n+1)*(n+2) + k*(2*(n-1)*n+(pi)^2/(2*n-1)))/len;
end