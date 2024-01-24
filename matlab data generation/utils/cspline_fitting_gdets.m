mat=load('C:\research\SIMFLUX_benchmark\benchmark\simulation\estimator_benchmark\cspline_biplane\coeff.mat');
coeff1=mat.coeff1;
coeff2=mat.coeff2;


ddz=-200;
zarr=zeros([21,51]);
zzz=linspace(0,500,21);
for i=1:21
    theta=[0,0,zzz(i),5000,0];

rnew=dmu_fun(theta(1),theta(2),theta(3)/5.0,theta(4),theta(5),coeff1,coeff2);
img_ch1=rnew.mu1;
img_ch2=rnew.mu2;



    for j=1:51



init_theta=[0*(rand()-0.5),0*(rand()-0.5),0*(rand()-0.5)+zzz(i),50000,20];
theta=newton(init_theta,poissrnd(img_ch1),poissrnd(img_ch2),coeff1,coeff2);
zarr(i,j)=theta(3);
    end
end