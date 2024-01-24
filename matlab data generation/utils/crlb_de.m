function CRLB=crlb_de(z,I,bg,PSF,PSFderivatives,parameters)


zframe=1;
l=1;
if (zframe==1)
    parameters.zrange=[z,z];
end
im=PSF;
parameters.numparams=5;
parameters.fitmodel='xyz';
parameters.xemit = 0;
parameters.yemit = 0;
parameters.zemit = z;
parameters.Mx = size(im,1);
parameters.My = size(im,2);
parameters.Mz = zframe;
parameters.Ncfg = size(im,4);

parameters.xrange = parameters.pixelsize*parameters.Mx/2;
parameters.yrange = parameters.pixelsize*parameters.My/2;
parameters.signalphotoncount=I;
parameters.backgroundphotoncount=bg;


[F,CRLB] = get_fisher_crlb(PSF,PSFderivatives,parameters);