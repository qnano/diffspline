load('C:/research/issue1/simulator/p.mat');
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



n_z_slices=3;

parameters.lambda=690;


parameters.zemit =0;

parameters.zrange = [-750, 750];

parameters.Mx = 21 ;
parameters.My = 21 ;
parameters.Mz = 3;
parameters.Npupil=201;
parameters.pixelsize=5;

parameters.xrange = parameters.pixelsize*parameters.Mx/2;
parameters.yrange = parameters.pixelsize*parameters.My/2;
parameters.ztype='medium';

parameters.aberrations=aberrations;
parameters.numparams=5+length(aberrations);
parameters.xemit = 0;
parameters.yemit =0;
parameters.Mx = 21 ;
parameters.My = 21 ;

parameters.pixelsize=65;

parameters.xrange = parameters.pixelsize*parameters.Mx/2;
parameters.yrange = parameters.pixelsize*parameters.My/2;
fn="C:/research/spin disk/simulation/PSF/OTFandPSF.mat";
mat=load(fn);
[PSF_img] = get_PSFfromOTF_matrix(mat.OTF_em,parameters);
PSF_img=PSF_img(:,:,1);
fn="C:/research/spin disk/simulation/PSF/line_img_65nm_pixelsize_500nm_seperate.mat";
mat=load(fn);
img=mat.img;
img_dev = deconvlucy(img,PSF_img);