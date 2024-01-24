function [PSF,PSFderivatives] = get_psfs_derivatives(FieldMatrix,FieldMatrixDerivatives)
% This function calculates the free or fixed dipole PSFs given the field
% matrix, the dipole orientation, and the pupil polarization, as well as
% the derivatives w.r.t. the xyz coordinates of the emitter and w.r.t. the
% emission wavelength lambda.
%
% copyright Sjoerd Stallinga, TU Delft, 2017
%
% parameters: emitter/absorber dipole orientation (characterized by angles
% pola and azim), detection/illumination polarization in objective lens
% back aperture (characterized by angles alpha and beta).

% find dimensions and number of derivatives from input
dims = size(FieldMatrixDerivatives);
if (length(dims)>3)
  Mz = dims(3);
  numders = dims(4);
  imdims = size(FieldMatrix{1,1,1});
else
  Mz = 1;
  numders = dims(3);
  imdims = size(FieldMatrix{1,1});
end
Mx = imdims(1);
My = imdims(2);
%disp(numders);
% calculation of free and fixed dipole PSF and the derivatives for the focal stack
PSF = zeros(Mx,My,Mz);
%PPSFderivatives = zeros(Mx,My,Mz,numders);
PSFderivatives = zeros(Mx,My,Mz,numders);

aaa=(2.0/3);
for jz = 1:Mz
  
    for jtel = 1:3

        for itel = 1:2
          PSF(:,:,jz) = PSF(:,:,jz) + (1/3)*abs(FieldMatrix{itel,jtel,jz}).^2;
          for jder = 1:numders
              
                  
                  PSFderivatives(:,:,jz,jder) = PSFderivatives(:,:,jz,jder) +...
                      aaa*real(conj(FieldMatrix{itel,jtel,jz}).*FieldMatrixDerivatives{itel,jtel,jz,jder});
              
          end
        end
      
    end
  
 


%PSF=PSF(2:end-1,2:end-1);
%PSFderivatives =PPSFderivatives(2:end-1,2:end-1,:,:);
end

