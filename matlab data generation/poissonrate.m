function [mu,dmudtheta] = poissonrate(theta,parameters)
% Returns the Poisson-rates for all pixels and all first order
% derivatives w.r.t. the parameters theta.
%
% copyright Sjoerd Stallinga, TU Delft, 2017

numparams = length(theta);
Mx = parameters.Mx;
My = parameters.My;
Mz = parameters.Mz;
parameters.xemit = theta(1);
parameters.yemit = theta(2);
Nph = theta(numparams-1);
bg = theta(numparams);
switch parameters.fitmodel
  case 'xyz'
    parameters.zemit = theta(3);
  case 'xylambda'
    parameters.lambda = theta(3);
  case 'xyzlambda'
    parameters.zemit = theta(3);
    parameters.lambda = theta(4);
  case 'aberrations'
    parameters.zemit = theta(3);
    parameters.aberrations(:,3) = theta(4:numparams-2); 
case 'aberrationsamp'
    numparams_phaseaberr=parameters.numparams_phaseaberr;
    numparams_ampaberr=parameters.numparams_ampaberr;
    parameters.zemit = theta(3);
    parameters.aberrations(4:end,3) = theta(4:4+numparams_phaseaberr-1); 
    parameters.aberrations(:,4) = theta(4+numparams_phaseaberr:4+numparams_phaseaberr+numparams_ampaberr-1); 
end

% Evaluate PSF and PSF derivatives
[~,~,wavevector,wavevectorzimm,Waberration,allzernikes,PupilMatrix] = ...
  get_pupil_matrix(parameters);

parameters.PupilMatrix=PupilMatrix;
parameters.allzernikes=allzernikes;
parameters.Waberration=Waberration;
parameters.wavevectorzimm=wavevectorzimm;
parameters.wavevector=wavevector;


[FieldMatrix,FieldMatrixDerivatives] = ...
  get_field_matrix_derivatives_fast(parameters.PupilMatrix,parameters.wavevector,parameters.wavevectorzimm,parameters.xemit,parameters.yemit,parameters.zemit,parameters.Ax,parameters.Bx,parameters.Dx);

[PSF,PSFderivatives] = get_psfs_derivatives(FieldMatrix,FieldMatrixDerivatives);
% get Poisson rate and derivatives
PSF = PSF/sum(PSF, 'all');
mu = Nph*PSF+bg;

% get derivatives of Poisson rate w.r.t. fit parameters
% if z-dimension is singleton we must make a difference
if Mz==1
  dmudtheta = zeros(Mx,My,numparams);
  for jp = 1:numparams-2
      %disp(size(PSFderivatives));
    dmudtheta(:,:,jp) = Nph*PSFderivatives(:,:,1,jp);
  end
  dmudtheta(:,:,numparams-1) = PSF;
  dmudtheta(:,:,numparams) = ones(size(PSF));
else
  dmudtheta = zeros(Mx,My,Mz,numparams);
  for jp = 1:numparams-2
    dmudtheta(:,:,:,jp) = Nph*PSFderivatives(:,:,:,jp);
  end
  dmudtheta(:,:,:,numparams-1) = PSF;
  dmudtheta(:,:,:,numparams) = ones(size(PSF));
end

end