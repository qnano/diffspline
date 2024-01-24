function dataout = cztfunc_v1(datain,xsize,qsize,NN,MM)
% This function evaluates the FT via the czt-algorithm
% arguments: datain = input data, dimensions K x N
%            A,B,D = auxiliary vectors computed in prechirpz, must have
%            lengths N, M, and L=N+M-1
% function value: dataout = output data, dimensions K x M
%
% copyright Sjoerd Stallinga, TU Delft, 2017
compl = [2+5i 13i];

L = NN+MM-1;
sigma = 2*pi*xsize*qsize/NN/MM;
Afac = exp(2*1i*sigma*(1-MM));
Bfac = exp(2*1i*sigma*(1-NN));
sqW = exp(2*1i*sigma);
W = sqW^2;
Gfac = (2*xsize/NN)*exp(1i*sigma*(1-NN)*(1-MM));

Utmp = zeros(1,NN,'like', compl);
A = zeros(1,NN,'like', compl);
Utmp(1) = sqW*Afac;
A(1) = 1.0;
for i=2:NN
  A(i) = Utmp(i-1)*A(i-1);
  Utmp(i) = Utmp(i-1)*W;
end
  
Utmp = zeros(1,MM,'like', compl);
B = ones(1,MM,'like', compl);
Utmp(1) = sqW*Bfac;
B(1) = Gfac;
for i=2:MM
  B(i) = Utmp(i-1)*B(i-1);
  Utmp(i) = Utmp(i-1)*W;
end

Utmp = zeros(1,max(NN,MM)+1,'like', compl);
Vtmp = zeros(1,max(NN,MM)+1,'like', compl);
Utmp(1) = sqW;
Vtmp(1) = 1.0;
for i=2:max(NN,MM)+1
  Vtmp(i) = Utmp(i-1)*Vtmp(i-1);
  Utmp(i) = Utmp(i-1)*W;
end
D = ones(1,L,'like', compl);
for i=1:MM
  D(i) = conj(Vtmp(i));
end
for i=1:NN
  D(L+1-i) = conj(Vtmp(i+1));
end
  
D = fft(D);

%%%%%%%%%%%%%
N = length(A);
M = length(B);
L = length(D);
K = size(datain,1);
Amt = repmat(A,K,1);
Bmt = repmat(B,K,1);
Dmt = repmat(D,K,1);
compl = [2+5i 13i];
cztin =  zeros(K,L,'like', compl);
cztin(:,1:N)= Amt.*datain;
temp = Dmt.*fft(cztin,[],2);
cztout = ifft(temp,[],2);
dataout = Bmt.*cztout(:,1:M);
  
end

