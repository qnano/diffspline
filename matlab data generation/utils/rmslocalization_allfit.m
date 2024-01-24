clear all
close all
maxn=5;
count=1;
for i=1:maxn
    mind=-i:2:i;
    for j=1:length(mind)
        aberrations(count,:)=[i,mind(j),0,0];
        count=count+1;
    end
end
aberrations(3,2)=0;
aberrations(4,2)=-2;
aberrationsall=aberrations;
aberrations_amp=aberrations;
aberrations_phase=aberrations;

filepath="C:/code/ExamplePhaseRetrieval/data/rmsdata/calib36rms_nobead.mat";
mat=load(filepath);
parameters=mat.para;

fn="C:/code/ExamplePhaseRetrieval/data/rmsdata/36rms_nobead.mat";
mat=load(fn);
im=(mat.imgl-3)./5e3;
im=im(:,:,5:7);
parameters.fitmodel='aberrationsamp';
parameters.zrange=[-200,200];
parameters.Nitermax=120;
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


parameters.Mx = size(im,1);
parameters.My = size(im,2);
parameters.Mz = size(im,3);
parameters.Ncfg = size(im,4);

parameters.xrange = parameters.pixelsize*parameters.Mx/2;
parameters.yrange = parameters.pixelsize*parameters.My/2;
err=zeros(10,64);
Iall=zeros(10,1);
for i=1:10
    
    Iall(i)=1e4*(0.7)^i;
    
end

for ii=1:10
    for i=1:64
        TFSuncorexp=poissrnd(im.*Iall(ii)+3);

        [~,~,wavevector,wavevectorzimm,Waberration,allzernikes,PupilMatrix] = get_pupil_matrix(parameters);

        [XImage,YImage,~,~] = get_field_matrix_derivatives(PupilMatrix,wavevector,wavevectorzimm,Waberration,allzernikes,parameters);

        % MLE fit
        thetainit = initialvalues(TFSuncorexp,XImage,YImage,parameters);
        [thetastore,TFSuncorfit_phase,~,~] = localization(TFSuncorexp,thetainit,parameters);
        thetauncor = squeeze(thetastore(:,:,end));
        err(ii,i)=thetauncor(1);

    end
end

crlbarr=zeros(10,1);
fn="C:/code/ExamplePhaseRetrieval/data/rmsdata/36rms_nobead.mat";
mat=load(fn);
parameters=mat.parameters;
parameters.xemit = 0;
parameters.yemit = 0;
parameters.zemit = 0;
parameters.numparams=5;
parameters.fitmodel='xyz';
for i=1:10
    parameters.signalphotoncount=Iall(i);
    parameters.backgroundphotoncount=3;

    [~,~,wavevector,wavevectorzimm,Waberration,allzernikes,PupilMatrix] = ...
          get_pupil_matrix(parameters);
    [~,~,FieldMatrix,FieldMatrixDerivatives] = ...
      get_field_matrix_derivatives(PupilMatrix,wavevector,wavevectorzimm,Waberration,allzernikes,parameters);

    [PSF,PSFderivatives] = get_psfs_derivatives(FieldMatrix,FieldMatrixDerivatives,parameters);

    [F,CRLB] = get_fisher_crlb(PSF,PSFderivatives,parameters);
    crlbarr(i,1)=CRLB(1);
end

plot(Iall,sqrt(mean(err.*err,2)));
hold on
plot(Iall,crlbarr,"--");
legend({'pecision','CRLB'})
xlabel("intensity(photons)");
ylabel("\sigma_{c}");
set(gca,'XScale','log','YScale','log')