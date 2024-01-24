clear all
a=vector_fitter_class;
filepath="C:/code/ExamplePhaseRetrieval/data/rmsdata/calib36rms_nobead.mat";
mat=load(filepath);
parameters=mat.para;

a.parameters=parameters;
numparams_phaseaberr=length(parameters.aberrations)-3;
numparams_ampaberr=length(parameters.aberrations);
a.parameters.numparams=5+numparams_ampaberr+numparams_phaseaberr;
a.parameters.numparams_phaseaberr=numparams_phaseaberr;
a.parameters.numparams_ampaberr=numparams_ampaberr;

parameters.xemit=40;
parameters.yemit=40;


a.parameters=parameters;
numparams_phaseaberr=length(parameters.aberrations)-3;
numparams_ampaberr=length(parameters.aberrations);
a.parameters.numparams=5+numparams_ampaberr+numparams_phaseaberr;
a.parameters.numparams_phaseaberr=numparams_phaseaberr;
a.parameters.numparams_ampaberr=numparams_ampaberr;

a.parameters.zrange=[-1000.0,1000.0];

Iall=zeros(15,1);
for i=1:15
    Iall(i)=1e5*(0.7)^i;
end
z=linspace(-1000,1000,11);
bg=3;

fn="C:/code/ExamplePhaseRetrieval/data/rmsdata/18rms_nobead.mat";
mat=load(fn);
im=(mat.imgl-3)./5e3;
im=im(:,:,:);
CRLBarr=zeros(15,11);
for i=1:15
    for zi=6:6
        I=Iall(i);
        bg=3;
        x=0;
        y=0;
        
        [F,CRLB,PSF]=a.get_crlb(x,y,z(zi),I,bg,im,15);
        CRLBarr(i,zi)=CRLB(1);
    end
end

save('C:/code/ExamplePhaseRetrieval/data/rmsdata/36rmscrlb_nobead_15frames_all','CRLBarr');
