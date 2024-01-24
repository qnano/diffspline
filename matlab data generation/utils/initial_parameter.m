function [parameters]=initial_parameter()

Npupil=201;
mat=load('C:/research/issue1/simulator/p.mat');
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
n_pixels =512;
n_z_slices=3;
parameters.lambda=640;
parameters.pixelsize=5;
parameters.NA=1.3;
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

parameters.aberrations=aberrations;
parameters.numparams=5+length(aberrations);


parameters.xemit = 0;
parameters.yemit =0;



end