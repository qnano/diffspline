function [XPupil,YPupil,wavevector,wavevectorzimm,Waberration,allzernikes,PupilMatrix] =...
  get_pupil_matrix(parameters)
% This function calculates the pupil matrix Q_{jk}, which gives the j-th
% electric field component proportional to the k-th dipole vector
% component.
%
% copyright Sjoerd Stallinga, TU Delft, 2017

% parameters: NA, refractive indices of medium, cover slip, immersion fluid,
% wavelength (in nm), sampling in pupil
NA = parameters.NA;
refmed = parameters.refmed;
refcov = parameters.refcov;
refimm = parameters.refimm;
refimmnom = parameters.refimmnom;
lambda = parameters.lambda;
Npupil = parameters.Npupil;

% pupil radius (in diffraction units) and pupil coordinate sampling
PupilSize = 1.0;
DxyPupil = 2*PupilSize/Npupil;
XYPupil = -PupilSize+DxyPupil/2:DxyPupil:PupilSize;
[YPupil,XPupil] = meshgrid(XYPupil,XYPupil);

% calculation of relevant Fresnel-coefficients for the interfaces
% between the medium and the cover slip and between the cover slip
% and the immersion fluid
% The Fresnel-coefficients should be divided by the wavevector z-component
% of the incident medium, this factor originates from the
% Weyl-representation of the emitted vector spherical wave of the dipole.
% bug correction 20180402
% 
% Correction factor moved to aplanatic amplitude factor, MSiemons 20180405

CosThetaMed = sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refmed^2);
CosThetaCov = sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refcov^2);
CosThetaImm = sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refimm^2);
CosThetaImmnom = sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refimmnom^2);
FresnelPmedcov = 2*refmed*CosThetaMed./(refmed*CosThetaCov+refcov*CosThetaMed);
FresnelSmedcov = 2*refmed*CosThetaMed./(refmed*CosThetaMed+refcov*CosThetaCov);
FresnelPcovimm = 2*refcov*CosThetaCov./(refcov*CosThetaImm+refimm*CosThetaCov);
FresnelScovimm = 2*refcov*CosThetaCov./(refcov*CosThetaCov+refimm*CosThetaImm);
FresnelP = FresnelPmedcov.*FresnelPcovimm;
FresnelS = FresnelSmedcov.*FresnelScovimm;

% setting of vectorial functions
Phi = atan2(YPupil,XPupil);
CosPhi = cos(Phi);
SinPhi = sin(Phi);
CosTheta = sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refmed^2);      
SinTheta = sqrt(1-abs(CosTheta).^2);

pvec{1} = FresnelP.*CosTheta.*CosPhi;
pvec{2} = FresnelP.*CosTheta.*SinPhi;
pvec{3} = -FresnelP.*SinTheta;
svec{1} = -FresnelS.*SinPhi;
svec{2} = FresnelS.*CosPhi;
svec{3} = 0;

PolarizationVector = cell(2,3);
for jtel = 1:3
  PolarizationVector{1,jtel} = CosPhi.*pvec{jtel}-SinPhi.*svec{jtel};
  PolarizationVector{2,jtel} = SinPhi.*pvec{jtel}+CosPhi.*svec{jtel};
end

% definition aperture
ApertureMask = double((XPupil.^2+YPupil.^2)<1.0);

% aplanatic amplitude factor
% combining this factor with the Fresnel-coefficient factors T_{p} and T_{s}
% the amplitude scales as [sqrt(cos(theta_imm))/(cos(theta_med))] x T_{p,s}
% where T_{p} = T_{p,med->cov} x T_{p,cov->imm}
% and T_{s} = T_{s,med->cov} x T_{s,cov->imm}
% with T_{p,med->cov} = 2*ref_med*cos(theta_med)/[ref_med*cos(theta_cov)+ref_cov*cos(theta_med)]
% and T_{s,med->cov} = 2*ref_med*cos(theta_med)/[ref_med*cos(theta_med)+ref_cov*cos(theta_cov)]
% in case of index matching the overall amplitude scaling is with
% [1/sqrt(cos(theta_med))] x T_{p,s}
% as the code previously expressed
% bug correction 20180402
% 
% NOTES MSiemons 20180409
% Implementation actually scaled as [sqrt(cos(theta_imm))./(refmed*cos(theta_med))]x T_{p,s}.
% Now all amplitude factors are added in the line below, for clarity.

Amplitude = ApertureMask.*sqrt(CosThetaImm)./(refmed*CosThetaMed);

% calculation aberration offset function
if parameters.aberrationcorrected
  beginindex = 1;
  orders_offset = parameters.aberrationsoffset(beginindex:end,1:2);
  zernikecoefs_offset = squeeze(parameters.aberrationsoffset(beginindex:end,3));
  normfac = sqrt(2*(orders_offset(:,1)+1)./(1+double(orders_offset(:,2)==0)));
  zernikecoefs_offset = normfac.*zernikecoefs_offset;
  allzernikes_offset = get_zernikefunctions(orders_offset,XPupil,YPupil);
  Waberration = zeros(size(XPupil));
  for j = 1:numel(zernikecoefs_offset)
    Waberration = Waberration+zernikecoefs_offset(j)*squeeze(allzernikes_offset(j,:,:));  
  end
%   Waberration = Waberration*lambda;
else
  Waberration = zeros(size(XPupil));
end

% calculation aberration function native to optical system
% Waberration = zeros(size(XPupil));
orders = parameters.aberrations(:,1:2);
zernikecoefs = squeeze(parameters.aberrations(:,3));
normfac = sqrt(2*(orders(:,1)+1)./(1+double(orders(:,2)==0))); 
%normfac=1.0;
zernikecoefs = normfac.*zernikecoefs;
% allzernikes is normalized
allzernikes = get_zernikefunctions(orders,XPupil,YPupil);
for j = 1:numel(zernikecoefs)
  Waberration = Waberration+zernikecoefs(j)*squeeze(allzernikes(j,:,:));  
end

% calculation DOE/SLM phase function
if ~strcmp(parameters.doetype,'none')
  orders_zone = parameters.zonefunction(:,1:2);
  zernikecoefs_zone = squeeze(parameters.zonefunction(:,3));
  normfac = sqrt(2*(orders_zone(:,1)+1)./(1+double(orders_zone(:,2)==0)));
  zernikecoefs_zone = normfac.*zernikecoefs_zone;
  allzernikes_zone = get_zernikefunctions(orders_zone,XPupil,YPupil);
  ZoneFunction = zeros(size(XPupil));
  for j = 1:numel(zernikecoefs_zone)
    ZoneFunction = ZoneFunction+zernikecoefs_zone(j)*squeeze(allzernikes_zone(j,:,:));  
  end
 

end

% compute effect of refractive index mismatch, depending on SAF conditions
% zvals = [nominal stage position, free working distance, -image depth from cover slip]
if and(NA>parameters.refmed, parameters.depth< 4*lambda)
    [zvals,~] = get_saffocus(parameters,PolarizationVector,Amplitude,Waberration);
else
    [zvals,~] = get_rimismatchpars(parameters);
end

Waberration = Waberration+zvals(1)*refimm*CosThetaImm-zvals(2)*refimmnom*CosThetaImmnom-zvals(3)*refmed*CosThetaMed;

Waberration = Waberration.*ApertureMask;
PhaseFactor = exp(2*pi*1i*Waberration/lambda);


if strcmp(parameters.fitmodel,'aberrationsamp') 
    AMPaberration = zeros(size(XPupil));
    for j = 1:length(parameters.aberrations)
        normfac = sqrt(2*(orders(j,1)+1)./(1+double(orders(j,2)==0)));
        %normfac = 1;
        AMPaberration = AMPaberration+normfac*parameters.aberrations(j,4)*squeeze(allzernikes(j,:,:));
    end
    AMPaberration=AMPaberration.*ApertureMask;
end

if strcmp(parameters.fitmodel,'aberrationsamp') 
    PupilMatrix = cell(2,3);
    for itel = 1:2
      for jtel = 1:3
        PupilMatrix{itel,jtel} = exp(-AMPaberration) .*Amplitude.*PhaseFactor.*PolarizationVector{itel,jtel};
      end
    end
else
% compute pupil matrix
    PupilMatrix = cell(2,3);
    for itel = 1:2
      for jtel = 1:3
        PupilMatrix{itel,jtel} = Amplitude.*PhaseFactor.*PolarizationVector{itel,jtel};
      end
    end
end

% calculate intensity normalization by flow of energy through lens aperture
[normint_free,normint_fixed] = get_normalization(PupilMatrix,parameters);
for itel = 1:2
  for jtel = 1:3
    if strcmp(parameters.dipoletype,'free')
      PupilMatrix{itel,jtel} = PupilMatrix{itel,jtel}/sqrt(normint_free);
    end
    if strcmp(parameters.dipoletype,'fixed')
      PupilMatrix{itel,jtel} = PupilMatrix{itel,jtel}/sqrt(normint_fixed);
    end
  end
end

% calculate wavevector inside immersion fluid and z-component inside medium 
wavevector = cell(1,3);
wavevector{1} = (2*pi*NA/lambda)*XPupil;
wavevector{2} = (2*pi*NA/lambda)*YPupil;
wavevector{3} = (2*pi*refmed/lambda)*CosThetaMed;
wavevectorzimm = (2*pi*refimm/lambda)*CosThetaImm; 

% plotting intermediate results
if parameters.debugmode
  figure
  for itel = 1:2
    for jtel = 1:3
      tempim = PupilMatrix{itel,jtel};
      subplot(2,3,3*(itel-1)+jtel)
      imagesc(rot90(abs(tempim)))
      title(strcat('amplitude i=',num2str(itel),', j=',num2str(jtel)))
      axis square
      axis off
%       colorbar
      caxis([0 0.08]);
    end
  end
  figure
  for itel = 1:2
    for jtel = 1:3
      tempim = PupilMatrix{itel,jtel};
      subplot(2,3,3*(itel-1)+jtel)
      phase = angle(tempim);
      UAFmask = (imag(wavevector{3}) == 0);
      Waberration(~ApertureMask) = nan;
      phase = imag(Waberration/lambda*2*pi);
%       phase = phase + angle(Amplitude);
      imagesc(phase-min(phase(:)))
      title(strcat('phase i=',num2str(itel),', j=',num2str(jtel)))
      axis square
      axis off
%       caxis([0 pi]);
      colorbar
    end
  end
end

end