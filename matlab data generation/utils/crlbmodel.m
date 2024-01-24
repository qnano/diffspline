clear all
%%%model%%%%%
a=vector_fitter_class;
fn="C:/code/ExamplePhaseRetrieval/data/figure2/18rms_astig50_2304.mat";
mat=load(fn);
parameters=mat.parameters;
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
parameters.aberrations=aberrations;
parameters.aberrations(:,3:4)=0;
parameters.aberrations(5,3)=50;
parameters.pixelsize=65;
%parameters.samplingdistance = parameters.pixelsize;
parameters.Nitermax=120;
a.parameters=parameters;
numparams_phaseaberr=length(parameters.aberrations)-3;
numparams_ampaberr=length(parameters.aberrations);
a.parameters.numparams=5+numparams_ampaberr+numparams_phaseaberr;
a.parameters.numparams_phaseaberr=numparams_phaseaberr;
a.parameters.numparams_ampaberr=numparams_ampaberr;
%%%%%%%%%%%
fn="C:/code/jelmer/data/0109-2/gataquant-bead-astig13-"+num2str(1)+".tif";
%fn="C:/code/jelmer/data/0109-2/flat-"+num2str(img_ind)+".tif";
tiff_info = imfinfo(fn); % return tiff structure, one element per image
tiff_stack = imread(fn, 1) ; % read in first image
z=linspace(-3000,3000,51);
crlbz=zeros(51,1);
for i=1:51
    [f,crlb,PSF]=a.get_crlb(0,0,z(i),1.0e3,3,tiff_stack,1);
    crlbz(i,1)=crlb(3);
end
plot(z,crlbz,"*--");
hold on
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
parameters.aberrations=aberrations;
parameters.aberrations(:,3:4)=0;
parameters.aberrations(13,3)=100;
a.parameters=parameters;
crlbz=zeros(51,1);
for i=1:51
    [f,crlb,PSF]=a.get_crlb(0,0,z(i),1.0e3,3,tiff_stack,1);
    crlbz(i,1)=crlb(3);
end
plot(z,crlbz,"+--");

legend({'astigmatism','tetrapod'});