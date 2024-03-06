 function [model,it] = SemismoothNewtonTrain(A,b,C)
%% Train L2-loss L2-regularized linear SVM by Semismooth Newton method
%This code is to use SemismoothNewtonTrain.m  
% to  solve binary classification problem 
%with L2-loss L2-regularized linear SVM by Semismooth Newton method.
% it is based on the following paper
% Yin J. and Li Q.N.*, A Semismooth Newton Method for Support Vector Classification 
%and Regression, Computational Optimization and Applications., 2019, 73(2) ï¼? 477-508
% last modified by Juan Yin and Qingna Li 2020/08/09 
% If you have comments, please contact qnl@bit.edu.cn

%Input:
% A: d-by-n training data matrix 
%    (n is number of data points and d is dimension of one data point)
% b: n-by-1 training labels vector
%    (each label is either -1 or 1)
% C: cost parameter of the support vector machine
%
%Output:
% model: 
%   model.weight: (d+1)-by-1 weight vector 
%   model.C: cost parameter of the support vector machine

format short
t0 = clock;
[~, n] = size(A);
A = [A; ones(1, n)];
B=A;
idx=find(b<0);
B(:,idx)=-A(:,idx);
[m,~]=size(A);
weight=ones(m,1);
maxk=100;
rho=0.5;  
sigma=1.0e-4;
eta0=0.05;
eta1=0.5;
maxit=200;
tol=1e-2;
k=0;
epsilon=1.0e-6;
cg_time = 0;
[ f0,c,idx1]=fun(B,weight,C);
g=gradient(B,weight,C,c);
normg = norm(g);
totalcgiter = 0;

while(k<maxk && normg >epsilon)
    muk=min(eta0,eta1*normg);
    rhs=-g;
    cg_time0=clock;
    Asub = A(:,idx1);
    [d,~,~,iterk] = cg(Asub, m,C,tol,maxit,rhs,muk,normg);
    tmp1=Jacobian(Asub,C,d);
    tmp2=norm(tmp1+g);
    fprintf('Jd+g=%f\n',tmp2);
    totalcgiter = totalcgiter+iterk;
    cg_time = cg_time + etime(clock,cg_time0);
    mk=0;
    alpha=rho^mk;
    descent = sum(g.*d);
    y = weight + alpha*d;
    [f,c,idx1]=fun(B,y,C);
    while (mk<10 && f>f0+sigma*alpha*descent+1.0e-4)
        mk= mk+1;
        alpha=rho^mk;
        y = weight+ alpha*d;
        [ f,c,idx1]=fun(B,y,C);
    end 
    weight = y;
    g=gradient(B,weight,C,c);
    f0=f;
    normg = norm(g);
    fprintf('k = %d, alpha = %1.1f, cg iteration number = %d  norm(g)=%f, t= %f\n ',k,alpha, iterk,normg, etime(clock,t0))
    k=k+1;
end


time_used = etime(clock,t0);
fprintf('Newton-CG: computing time for linear systems solving (cgs time) ==== %.4f (s)\n', cg_time)
fprintf('Newton-CG: computing time used for the total program ===== %.4f (s)\n',time_used)

model.weight = weight;
model.C = C;
it=  totalcgiter;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% subfunctions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [d,flag,relres,iterk] = cg(Asub, m,C,tol,maxit,rhs,~,normg)
% Initializations
r = rhs;  %We take the initial guess x0=0 to save time in calculating A(x0) 
n2b =normg;    % norm of D
tolb = tol * n2b;  % relative tolerance 
d = zeros(m,1);
flag=1;
iterk =0;
relres=1000; %%% To give a big value on relres
% Precondition 
z =r;  
rz1 = r'*z; 
rz2 = 1; 
p = z;
% CG iteration
for k = 1:maxit
   if k > 1
       beta = rz1/rz2;
       p = z + beta*p;
   end
   w = Jacobian(Asub,C,p) ;
   denom = p'*w;
   iterk =k;
   relres = norm(r)/n2b;              %relative residue = norm(r) / norm(b)
   if denom <= 0 
       d = p/norm(p); % p is not a descent direction
       break % exit
   else
       alpha = rz1/denom;
       d = d + alpha*p;
       r = r - alpha*w;
   end
   z = r; 
   if norm(r) <= tolb % Exit if Hp=b solved within the relative tolerance
       iterk =k;
       relres = norm(r)/n2b;      %relative residue =norm(r) / norm(b)
       flag =0;
       break
   end
   rz2 = rz1;
   rz1 = r'*z;
end

%% objective function
function[ f,c,idx1]=fun(B,weight,C)
tmp=1-B'*weight;
c=max(0,tmp);
idx1=find(c);
c1=c(idx1);
 
f=0.5*sum(weight.*weight);
f=f+C*sum(c1.*c1);
return

%% gradient function
function g=gradient(B,weight,C,c)
tmp = B*c;
g = -2*C*tmp;
g=weight+g;
return

%% generalized jacobian function
function  vs=Jacobian(Asub,C,p)
tmp=Asub'*p;
tmp1=Asub*tmp;
vs=(1+1.0e-10)*p+ 2*C*tmp1;
  
  
  
  