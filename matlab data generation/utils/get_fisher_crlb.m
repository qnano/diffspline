function [FisherMatrix,CRLB] = get_fisher_crlb(PSF,PSFderivatives,parameters)
% This function is for calculating the Fisher matrix and the CRLB given the
% exact vectorial PSF model. The parameters depend on the fitmodel but are
% at most [x,y,z,lambda,N,b] with x,y,z, the coordinates of the emitter,
% lambda the wavelength, N the signal photon count and b the number of
% background photons per pixel. The script can work for a single ROI model:
% size(mu)=[Mx,My] or for multiple ROI model: size(mu)=[Mx,My,Mz].
%
% copyright Sjoerd Stallinga, TU Delft, 2017
%

numparams = parameters.numparams;
numders = numparams-2;
Nphindex = numparams-1;
bgindex = numparams;

keps = 1e3*eps;
Nph = parameters.signalphotoncount;
bg = parameters.backgroundphotoncount;
%rnvar = (parameters.readnoisestd)^2;
rnvar=0;
[Mx,My,Mz] = size(PSF);

% calculation of Poisson rates and derivatives
mu = Nph.*PSF+bg+rnvar;
mupos = double(mu>0).*mu + double(mu<0)*keps;
weight = 1./mupos;

dmudtheta = zeros(Mx,My,Mz,numparams);
if isfield(parameters,'bleaching')
    for jp = 1:numders
    if ~strcmp(parameters.fitmodel, 'xylambda') && jp == 3
        for ii = 1:parameters.Mz
            dmudtheta(:,:,ii,3) = Nph(1)*PSFderivatives(:,:,ii,jp)+bg(1)*parameters.bleachingderivatives(ii);
        end
    else
        dmudtheta(:,:,:,jp) = Nph(1)*PSFderivatives(:,:,:,jp);
    end
    end
    for ii = 1:parameters.Mz
        dmudtheta(:,:,ii,numparams-1) = parameters.bleaching(ii)*PSF(:,:,ii);
        dmudtheta(:,:,ii,numparams) = parameters.bleaching(ii)*ones(size(PSF(:,:,ii)));
    end
else
    dmudtheta(:,:,:,1:numders) = Nph*PSFderivatives(:,:,:,1:numders);
    dmudtheta(:,:,:,Nphindex) = PSF;
    dmudtheta(:,:,:,bgindex) = ones(size(PSF));
end
  
% calculation Fisher matrix
FisherMatrix = zeros(numparams,numparams);
for ii = 1:numparams
  for jj = ii:numparams
    FisherMatrix(ii,jj) = sum(sum(sum(weight.*dmudtheta(:,:,:,ii).*dmudtheta(:,:,:,jj))));
    FisherMatrix(jj,ii) = FisherMatrix(ii,jj);
  end
end
 
% regularization Fisher-matrix in order to circumvent possibility for
% inverting ill-conditioned matrix
if (rcond(FisherMatrix)>keps)
%   CRLB = sqrt(diag(inv(FisherMatrix+keps*eye(size(FisherMatrix)))));
  CRLB = sqrt(diag(inv(FisherMatrix)));
else
  CRLB = zeros(numparams,1);
end
  
end

