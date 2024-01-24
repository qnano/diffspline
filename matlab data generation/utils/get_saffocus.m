function [zvals,Wrms] = get_saffocus(parameters,PolarizationVector,Amplitude,Waberration)
% This function finds the stage positions for optimal focus in SAF 
% conditions by optimizing the Strehl ratio. 
% The Strehl ratio is computed for 10 z-planes around an initial estimate of 
% the stage position (-1.25*refimm/refmed*depth). Then a second order
% polynomial fit is performed near the maximum to achieve subsampling
% z-precision.
% The Wrms due to the refractive index mismatch is then computed by 
% Wrms = lambda/(2*pi)*log(1/S(Wab + Wri|Wab)), with S(Wab|Wab + Wri) the ratio 
% of the maximum intensity of PSF with aberration Wab and the maximum intensity 
% of PSF with additional aberration Wab ánd refrective index mismatch Wri. 
% 
% Note: for depth = 0 (at the coverslip) the Wrms becomes negative as
% maximum intensity is no longer at z-stage position = 0;
%
% relevant z-positions:
% zvals = [nominal stage position, free working distance, -image depth from cover slip]

% Marijn Siemons, 16 04 2018

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
ApertureMask = double((XPupil.^2+YPupil.^2)<1.0);

CosThetaMed = sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refmed^2);
CosThetaImm = sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refimm^2);
CosThetaImmnom = sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refimmnom^2);

zvals = [0 parameters.fwd -parameters.depth];

Strehlnorm = 0;
for itel = 1:2
  for jtel = 1:3
        Strehlnorm = Strehlnorm + abs(sum(sum(Amplitude.*exp(2*pi*1i*(Waberration)/lambda).*PolarizationVector{itel,jtel}))).^2;
  end
end

Nz = 15;
zpos = linspace(-250,250,Nz);
Strehl = zeros(1,Nz);
for jz = 1:Nz
    zvals(1) = parameters.fwd - 1.25*refimm/refmed*parameters.depth + zpos(jz);
    Wzpos = zvals(1)*refimm*CosThetaImm-zvals(2)*refimmnom*CosThetaImmnom-zvals(3)*refmed*CosThetaMed; 
    Wzpos = Wzpos.*ApertureMask;
    for itel = 1:2
      for jtel = 1:3
            Strehl(jz) = Strehl(jz) + abs(sum(sum(Amplitude.*exp(2*pi*1i*(Wzpos + Waberration)/lambda).*PolarizationVector{itel,jtel}))).^2./Strehlnorm;
      end
    end
end
[~, indz] = max(Strehl);
if indz <= 3; indz = 3; elseif indz > (Nz-3); indz = Nz-3; end
zfit = polyfit(zpos(indz-2:indz+2),Strehl(indz-2:indz+2),2);
zvals(1) = parameters.fwd - 1.25*refimm/refmed*parameters.depth - zfit(2)/(2*zfit(1));

MaxStrehl = polyval(zfit,- zfit(2)/(2*zfit(1)));
Wrms = lambda/(2*pi)*log(1/MaxStrehl);

% output relevant numbers
if parameters.debugmode
  fprintf('image plane depth from cover slip = %4.0f nm\n',-zvals(3))
  fprintf('free working distance = %6.3f mu\n',1e-3*zvals(2))
  fprintf('nominal z-stage position = %6.3f mu\n',1e-3*zvals(1))
  fprintf('rms aberration due to RI mismatch = %4.1f mlambda\n',1e3*Wrms/parameters.lambda)
  
  figure
  plot(zpos - 1.25*refimm/refmed*parameters.depth,Strehl)
  xlabel('Stage position')
  ylabel('Strehl') 
end

end

