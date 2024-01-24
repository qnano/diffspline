function [r,parameters]=dz_gaussian_withoutSAF_astig(x,y,zr,Nph,bg,z_steps,z_start,z_finish, ROIsize, spherical1, spherical2)
Npupil=z_steps;
mat=load('p.mat');
parameters=mat.parameters;
%parameters.FTlambda=640;
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

aberrations(5,3) = 100;
aberrations(12,3) = spherical1;
aberrations(26,3) = spherical2;

n_pixels = ROIsize;
parameters.pixelsize=100;

n_z_slices=1;
parameters.lambda=680;

parameters.NA=1.3;
parameters.zemit =0;

parameters.zrange = [z_start, z_finish];

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
NA = parameters.NA;
lambda = parameters.lambda;
xemit = parameters.xemit;
yemit = parameters.yemit;
zemit = parameters.zemit;
xrange = parameters.xrange;
yrange = parameters.yrange;
Npupil = parameters.Npupil;
Mx = parameters.Mx;

PupilSize = 1.0;
ImageSizex = xrange*NA/lambda;

[Ax,Bx,Dx] = prechirpz(PupilSize,ImageSizex,Npupil,Mx);
parameters.Ax=Ax;
parameters.Bx=Bx;
parameters.Dx=Dx;
        

parameters.xemit = x;
parameters.yemit =y;
parameters.zemit =zr;
theta = [x,y,zr,Nph,bg];
[mu,dmudtheta] = poissonrate(theta,parameters);
r.mu=mu;
r.dmudtheta=dmudtheta;
end