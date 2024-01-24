ddz=200;

x=0;
y=0;
z=0;
zr=z;
Nph=2500;
bg=10;
r=dz_gaussian_withoutSAF(x,y,zr,Nph,bg);
img_ch1=poissrnd(r.mu);

x=0;
y=0;
zr=z+ddz;
Nph=2500;
bg=10;
r=dz_gaussian_withoutSAF(x,y,zr,Nph,bg);
img_ch2=poissrnd(r.mu);

init_theta=[30*(rand()-0.5),30*(rand()-0.5),30*(rand()-0.5)+z,5000,20];
theta=LM_alg(init_theta,img_ch1,img_ch2);