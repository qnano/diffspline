function out = gammln(xx)
% this function calculates log(Gamma(xx)) following the numerical recipes
% algorithm
cofs = [76.18009172947146, -86.50532032941677, 24.01409824083091, ...
        -1.231739572450155, 0.12086509738666179e-2, -0.5395239384953e-5];
stp = 2.5066282746310005;
x = xx;
y = x;
tmp = x+5.5;
tmp = (x+0.5)*log(tmp)-tmp;
ser = 1.000000000190015;
for j = 1:6
  y = y+1.0;
  ser = ser+cofs(j)/y;
end
out = tmp+log(stp*ser/x);