clear all
%%%model%%%%%
a=vector_fitter_class;
filepath="C:/code/ExamplePhaseRetrieval/data/rmsdata/calib18rms_nobead.mat";
mat=load(filepath);
parameters=mat.para;

a.parameters=parameters;
numparams_phaseaberr=length(parameters.aberrations)-3;
numparams_ampaberr=length(parameters.aberrations);
a.parameters.numparams=5+numparams_ampaberr+numparams_phaseaberr;
a.parameters.numparams_phaseaberr=numparams_phaseaberr;
a.parameters.numparams_ampaberr=numparams_ampaberr;
%%%%%%%%%%%

fn="C:/code/ExamplePhaseRetrieval/data/rmsdata/36rms_nobead.mat";
mat=load(fn);
im=(mat.imgl-3)./5e3;

Iall=zeros(15,1);
for i=1:15
    Iall(i)=1e5*(0.7)^i;
end

bg=3;
err=zeros(15,11,128);
for i=1:15
    for ii=1:128
        imfit=poissrnd(im.*Iall(i)+bg);
        for iii=6:6
            
            theta=a.VF_localization(squeeze(imfit(:,:,iii)));
            err(i,iii,ii)=theta(1);
        end
    end
end
save('C:/code/ExamplePhaseRetrieval/data/rmsdata/36rms-18rms_lerrmoremore_nobead','err');