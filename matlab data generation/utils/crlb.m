clear all

fn="C:/code/ExamplePhaseRetrieval/data/figure2/18rms_astig50_2304.mat";
mat=load(fn);

fn="C:/code/jelmer/data/zernike-0.85pupil-1.tif";
tiff_info = imfinfo(fn); % return tiff structure, one element per image
tiff_stack = imread(fn, 1) ; % read in first image
%concatenate each successive tiff to tiff_stack
for ii = 2 : size(tiff_info, 1)
    temp_tiff = imread(fn, ii);
    tiff_stack = cat(3 , tiff_stack, temp_tiff);
end
tiff_stack=double(tiff_stack);

tiff_stack=(tiff_stack-103.1)*0.49;

im=tiff_stack(:,:,19:27);

z=linspace(-2000,2000,41);
[Mx,My,Mz]=size(tiff_stack);
parameters=mat.parameters;
parameters.fitmodel='aberrations';
parameters.Mx=Mx;
parameters.My=My;
parameters.Mz=9;
parameters.pixelsize=81.25;
parameters.bead=true;
parameters.zrange=[-2000,2000];
parameters.beaddiameter=500;
parameters.zspread=[-500,500];
%start the vector fitter class
a=vector_fitter_class;
%set parameters
a.parameters=parameters;
%set max radial degree
a.maxn=8;
a.parameters.xrange = a.parameters.pixelsize*a.parameters.Mx/2;
a.parameters.yrange = a.parameters.pixelsize*a.parameters.My/2;
%set z range
%you also can set other parameters. you can refer the class code 
tiff_stack=double(tiff_stack);

count=1;
ma = 8;
for i=1:ma
    mind=-i:2:i;
    for j=1:length(mind)
        aberrations(count,:)=[i,mind(j),0,0];
        count=count+1;
    end
end
aberrations(3,2)=0;
aberrations(4,2)=-2;


%aberrations=abb;
aberrationsall=aberrations;
%aberrations_phase=aberrations;


a.parameters.fitmodel='aberrations';

a.parameters.Nitermax=40;
if strcmp(a.parameters.fitmodel,'aberrationsamp') 
    a.parameters.aberrations=aberrations;
    numparams_phaseaberr=length(aberrations)-3;
    numparams_ampaberr=length(aberrations);
    a.parameters.numparams=5+numparams_ampaberr+numparams_phaseaberr;
    a.parameters.numparams_phaseaberr=numparams_phaseaberr;
    a.parameters.numparams_ampaberr=numparams_ampaberr;
elseif strcmp(a.parameters.fitmodel,'aberrations') 
    aberrations=aberrations(4:end,1:3);
    a.parameters.aberrations=aberrations;
    a.parameters.numparams=5+length(aberrations);
end


%% Perform fit of uncorrected TFS

a.parameters.Mx = size(im,1);
a.parameters.My = size(im,2);
a.parameters.Mz = size(im,3);
a.parameters.Ncfg = 1;

a.parameters.xrange = a.parameters.pixelsize*a.parameters.Mx/2;
a.parameters.yrange = a.parameters.pixelsize*a.parameters.My/2;

for iii=1:size(im,4)
    %TFSuncorexp=poissrnd(im.*2e3+3);
    TFSuncorexp=squeeze(im(:,:,:,iii));
    [~,~,wavevector,wavevectorzimm,Waberration,allzernikes,PupilMatrix] = get_pupil_matrix(a.parameters);

    [XImage,YImage,~,~] = get_field_matrix_derivatives(PupilMatrix,wavevector,wavevectorzimm,Waberration,allzernikes,a.parameters);

    % MLE fit
    fprintf('Fitting uncorrected...\n')
    thetainit = initialvalues(TFSuncorexp,XImage,YImage,a.parameters);
    [thetastore,~,~,~] = localization(TFSuncorexp,thetainit,a.parameters);
    thetauncor = squeeze(thetastore(:,:,end));

    for i=4:length(thetauncor)-2
        aberrationsall(i,3)=aberrationsall(i,3)+thetauncor(i);
    end
end

aberrationsall(:,3)=aberrationsall(:,3)/size(im,4);
aberrations=aberrationsall;
a.parameters.aberrations=aberrations;
a.parameters.signalphotoncount=thetauncor(end-1);
a.parameters.backgroundphotoncount=thetauncor(end);

a.parameters.xemit = 0;
a.parameters.yemit = 0;
a.parameters.zemit = 0;


[~,~,wavevector,wavevectorzimm,Waberration,allzernikes,PupilMatrix] = ...
      get_pupil_matrix(a.parameters);
[~,~,FieldMatrix,FieldMatrixDerivatives] = ...
  get_field_matrix_derivatives(PupilMatrix,wavevector,wavevectorzimm,Waberration,allzernikes,a.parameters);

[PSF,PSFderivatives] = get_psfs_derivatives(FieldMatrix,FieldMatrixDerivatives,a.parameters);

[F,CRLB] = get_fisher_crlb(PSF,PSFderivatives,a.parameters);