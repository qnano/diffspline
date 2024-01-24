function r=dz_gaussian_withSAF(x,y,zr,Nph,bg)
Npupil=201;
mat=load('C:\research\SIMFLUX_benchmark\benchmark\simulation\p.mat');
parameters=mat.parameters;
parameters.FTlambda=640;
count=1;
ma = 6;
for i=1:ma
    mind=-i:2:i;
    for j=1:length(mind)
        aberrations(count,:)=[i,mind(j),0,0];
        count=count+1;
    end
end
aberrations(3,2)=0;
aberrations(4,2)=-2;


n_pixels =21;
n_z_slices=1;
parameters.lambda=680;
parameters.pixelsize=100;
parameters.NA=1.49;
parameters.zemit =0;

parameters.zrange = [-750, 750];

parameters.Mx = n_pixels ;
parameters.My = n_pixels ;
parameters.Mz = n_z_slices;
parameters.Npupil=Npupil;
parameters.xrange = parameters.pixelsize*parameters.Mx/2;
parameters.yrange = parameters.pixelsize*parameters.My/2;
parameters.ztype='medium';
parameters.polarization_excite='circular';  % circular or linear
parameters.fitmodel='xyz';
parameters.aberrations=aberrations;
parameters.numparams=5+length(aberrations);


parameters.xemit = x;
parameters.yemit = y;
parameters.zemit = zr;
theta = [x,y,zr,Nph,bg];
[mu,dmudtheta] = poissonrate(theta,parameters);
r.mu=mu;
r.dmudtheta=dmudtheta;
end