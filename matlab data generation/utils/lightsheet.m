load('C:/research/chisquareeval/data/Exp003_correctedaberrationsand60mlambdaadded_Greenbead_180nm.mat');

n_pixels =31;
hn=(n_pixels-1)/2;
n_z_slices =51;
%aberrations = [1,-1,0,0; 1,1,0.0,0; 2,0,0,0.0; 2,-2,0,0.0; 2,2,50.0,0.0];
count=1;
ma = 5;
for i=1:ma
    mind=-i:2:i;
    for j=1:length(mind)
        aberrations(count,:)=[i,mind(j),0,0];
        count=count+1;
    end
end
aberrations(3,2)=0;
aberrations(4,2)=-2;
Wrmsuncor=0;
stand=(Wrmsuncor*parameters.lambda/1000)^2;
a=rand(length(aberrations),1)-0.5;
a(1:3,1)=0;
a=sqrt(stand)*a/sqrt(sum(a.^2));
aberrations(15,3)=100;
aberrations(:,3)=aberrations(:,3)+a;
%Wrmsuncor = sqrt(sum(a.^2))./parameters.lambda*1000;

parameters.aberrations=aberrations;
%parameters.samplingdistance = parameters.pixelsize;

parameters.NA = 1.49;

parameters.xemit = 0;
parameters.yemit = 0;

parameters.pixelsize=58;
%parameters.samplingdistance = parameters.pixelsize;
parameters.fitmodel='aberrationsamp';
%aberrations = [1,-1,0,0; 1,1,0.0,0; 2,0,0,0.05; 2,-2,0,0.0; 2,2,50.0,0.01; 3,-1,0.0,0.0; 3,1,0.0,0; 4,0,0.0,0.1; 3,-3,-0.0,0; 3,3,0.0,0; 4,-2,0.0,0; 4,2,0.0,0; 5,-1,0.0,0; 5,1,0.0,0; 6,0,0.0,0; 4,-4,0.0,0; 4,4,0.0,0;  5,-3,0.0,0; 5,3,0.0,0;  6,-2,0.0,0; 6,2,0.0,0; 7,1,0.0,0; 7,-1,0.0,0; 8,0,0.0,0];
parameters.beaddiameter = 500;
parameters.lambda = 680;
parameters.lambdacentral=680;
parameters.lambdaspread=[680 680];
parameters.zrange = [-2000, 2000];
parameters.doetype='none';
parameters.dipoletype='free';
parameters.Mx = n_pixels ;
parameters.My = n_pixels ;
parameters.Mz = n_z_slices;

parameters.xrange = parameters.pixelsize*parameters.Mx/2;
parameters.yrange = parameters.pixelsize*parameters.My/2;

if strcmp(parameters.fitmodel,'aberrationsamp') 
    parameters.aberrations=aberrations;
    numparams_phaseaberr=length(aberrations)-3;
    numparams_ampaberr=length(aberrations);
    parameters.numparams=5+numparams_ampaberr+numparams_phaseaberr;
    parameters.numparams_phaseaberr=numparams_phaseaberr;
    parameters.numparams_ampaberr=numparams_ampaberr;
elseif strcmp(parameters.fitmodel,'aberrations') 
    aberrations=aberrations(4:end,1:3);
    parameters.aberrations=aberrations;
    parameters.numparams=5+length(aberrations);
end

[~,~,wavevector,wavevectorzimm,Waberration,allzernikes,PupilMatrix] = ...
  get_pupil_matrix(parameters);
[~,~,FieldMatrix,FieldMatrixDerivatives] = ...
  get_field_fourpi(PupilMatrix,wavevector,wavevectorzimm,Waberration,allzernikes,parameters);

[PSF,PSFderivatives] = get_psfs_derivatives(FieldMatrix,FieldMatrixDerivatives,parameters);
PSF=PSF*-7.8784e-05;
X=linspace(-1,1,31);
Z=linspace(-1,1,51);
[x,y,z] = meshgrid(X,X,Z);

h = slice(x,y,z,PSF,[-.5 0 0.5],[],[0]);
set(h,'EdgeColor','none',...
'FaceColor','interp',...
'FaceAlpha','interp');
alpha('color');

alphamap('rampdown');
%alphamap('increase',.1);
colormap hsv;