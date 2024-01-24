clear all
%%%model%%%%%
a=vector_fitter_class;
filepath="C:/code/ExamplePhaseRetrieval/data/rmsdata/calib18rms_nobead.mat";
mat=load(filepath);
parameters=mat.para;
parameters.aberrations=parameters.aberrations(1:5,:);
parameters.aberrations(:,3:4)=0;
parameters.aberrations(5,3)=50;
parameters.pixelsize=80;
%parameters.samplingdistance = parameters.pixelsize;
parameters.Nitermax=120;
a.parameters=parameters;
numparams_phaseaberr=length(parameters.aberrations)-3;
numparams_ampaberr=length(parameters.aberrations);
a.parameters.numparams=5+numparams_ampaberr+numparams_phaseaberr;
a.parameters.numparams_phaseaberr=numparams_phaseaberr;
a.parameters.numparams_ampaberr=numparams_ampaberr;
%%%%%%%%%%%

fn="C:/code/ExamplePhaseRetrieval/data/rmsdata/18rms_nobead.mat";
mat=load(fn);
im=(mat.imgl-3)./5e3;
err=zeros(64,1);
for ii=1:64
    imfit=poissrnd(im(:,:,6).*1e5+3);
    [theta,mu]=a.VF_localization((imfit(:,:)));
    err(ii,1)=theta(2);
end
stderr=std(err);
rmse=sqrt(mean(err.*err));
[crlb,PSF]=a.get_crlb(0,0,0,1.5e3,10);
