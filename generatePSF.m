function generatePSF(x, y, Nph, bg, z_start, z_finish,z_steps, ROIsize, spherical1, spherical2, output_filename)

addpath('matlab data generation');
addpath('matlab data generation/utils');
x=double(x); y=double(y);

z_start = double(z_start); %-500;
z_finish = double(z_finish); %500;
z_steps = double(z_steps); %101;
ROIsize = double(ROIsize); %34;
%Nph=5000;
%bg=0;

spherical1 = double(spherical1);
spherical2 = double(spherical2);


z=linspace(z_start,z_finish,z_steps);

PSF=zeros([ROIsize,ROIsize,z_steps]);
for i=1:z_steps
[r, parameters]=dz_gaussian_withoutSAF_astig(x,y,z(i),Nph,bg,z_steps,z_start,z_finish, ROIsize, spherical1, spherical2);
PSF(:,:,i)=r.mu;

end

save(output_filename,'PSF', 'parameters')

end