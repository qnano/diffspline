clear all
x0=zeros(1,17);
options = optimset('Display', 'iter','MaxIter',500);
%x = fminsearch(cost,x0,options);

best=x0;
cbest=cost(x0);
size=1;
marr=zeros(1000,1);
for i=1:100
    x=best;
    %m=randi(17);
    amp=size*(rand(1,17)-0.5);
    x=x+amp;
    c=cost(x);
    marr(i)=c;
    if c<cbest
        cbest=c;
        best=x;
    end
    if mod(i,100)==0
        plot(marr);
    end
end
