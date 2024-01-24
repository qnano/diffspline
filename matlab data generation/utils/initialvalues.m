function thetainit = initialvalues(allspots,XImage,YImage,parameters)
% This function provides initial values for the fit parameters by means of
% a centroid fit.
%
% copyright Sjoerd Stallinga, TU Delft, 2017
%
if (parameters.fitmodel~="aberrationsamp")
    numparams = parameters.numparams;
else
    numparams=5+parameters.numparams_phaseaberr+parameters.numparams_ampaberr;
end
Ncfg = parameters.Ncfg;
dims = size(allspots);
if Ncfg>1
  if length(dims)>3
    [Mx,My,Mz,Ncfg] = size(allspots);
  else
    [Mx,My,Ncfg] = size(allspots);
  end
end
if Ncfg==1
  if length(dims)>2
    [Mx,My,Mz] = size(allspots);
  else
    [Mx,My] = size(allspots);
    Mz=1;
  end
end

thetainit = zeros(numparams,Ncfg);
for jcfg = 1:Ncfg
  if Ncfg>1
    if length(dims)>3
      dummat = allspots(:,:,round((Mz+1)/2),jcfg);
    else
      dummat = allspots(:,:,jcfg);
    end
  end
  if Ncfg==1
    if length(dims)>2
      dummat = allspots(:,:,round((Mz+1)/2));
    else
      dummat = allspots;
    end
  end
  
  % background estimate from the median value of the rim pixels,
  % with a minimum of 1.
  rimpixels = zeros(2*Mx+2*My-4,1);
  rimpixels(1:Mx-1) = dummat(1:Mx-1,1);
  rimpixels(Mx:2*Mx-2) = dummat(2:Mx,My);
  rimpixels(2*Mx-1:2*Mx+My-3) = dummat(Mx,1:My-1);
  rimpixels(2*Mx+My-2:2*Mx+2*My-4) = dummat(1,2:My);
  bg = median(rimpixels);
  bg = max(bg,1);
% alternative bg estimation from the minimum of the pixels+1.
%   bg = min(min(dummat))+1;
  
  % estimate of signal photon count
  dummat = dummat-bg;
  dummat = max(dummat,0);
  Nph = sum(sum(dummat));
  
  % rough correction for photon flux outside ROI
  Nph = 1.5*Nph;
      
  % calculation of the moments of the intensity distribution and centroid
  % estimate of lateral position
  Momx = sum(sum(XImage.*dummat));
  Momy = sum(sum(YImage.*dummat));
  x0 = Momx/Nph;
  y0 = Momy/Nph;
  
  % estimate additional parameters depending on fit model
  % this part of the code contains several "magic numbers" that
  % are adapted to specific aberration or DOE parameters values
  switch parameters.fitmodel
    case 'xyz'
      mask = ones(Mx,My);
      if (My>Mx)
        mask(:,1:round((My-Mx)/2)) = 0;
        mask(:,My-round((My-Mx)/2)+1:My) = 0;
      end
      Momxx = sum(sum(XImage.^2.*dummat.*mask));
      Momyy = sum(sum(YImage.^2.*dummat.*mask));
      Momxy = sum(sum(XImage.*YImage.*dummat.*mask));
      Nphr = sum(sum(dummat.*mask));
      Axx = Momxx-Nphr*x0^2;
      Ayy = Momyy-Nphr*y0^2;
      Axy = Momxy-Nphr*x0*y0;
      z0 = 1250*Axy/(Axx+Ayy);
    case 'xylambda'
      lambda = parameters.lambdacentral;
      switch parameters.doetype
        case 'binary'
% estimation of lambda from distance between spots
%           ydiv = 0;
          ydiv = y0;
          dumright = dummat.*double(YImage>=ydiv);
          dumleft = dummat.*double(YImage<=ydiv);
          Nright = sum(sum(dumright));
          Nleft = sum(sum(dumleft));
          yright = sum(sum(YImage.*dumright))/Nright;
          yleft = sum(sum(YImage.*dumleft))/Nleft;
          lambda = 0.70*(yright-yleft);

% update x0 from different photon count left/right
          y0 = (yleft+yright)/2;
      end
    case 'xyzlambda'
      switch parameters.doetype
        case 'none'
          mask = ones(Mx,My);
          if (My>Mx)
            mask(:,1:round((My-Mx)/2)) = 0;
            mask(:,My-round((My-Mx)/2)+1:My) = 0;
          end
          Momxx = sum(sum(XImage.^2.*dummat.*mask));
          Momyy = sum(sum(YImage.^2.*dummat.*mask));
          Momxy = sum(sum(XImage.*YImage.*dummat.*mask));
          Nphr = sum(sum(dummat.*mask));
          Axx = Momxx-Nphr*x0^2;
          Ayy = Momyy-Nphr*y0^2;
          Axy = Momxy-Nphr*x0*y0;
          z0 = 1250*Axy/(Axx+Ayy);
          lambda = parameters.lambdacentral;
%           lambda = 1.5*sqrt((Axx+Ayy-2*abs(Axy))/Nphr);
%           lambda = min(lambda,parameters.lambdacentral+parameters.lambdaspread);
%           lambda = max(lambda,parameters.lambdacentral-parameters.lambdaspread);
        case 'blazed'
% ratiometric estimation of lambda
          ydiv = 0;
          ydiv = y0;
          dumright = dummat.*double(YImage>=ydiv);
          dumleft = dummat.*double(YImage<=ydiv);
          Nright = sum(sum(dumright));
          Nleft = sum(sum(dumleft));
          lambda = parameters.lambdacentral+250*(Nright-Nleft)/(Nright+Nleft);
          yright = sum(sum(YImage.*dumright))/Nright;
          yleft = sum(sum(YImage.*dumleft))/Nleft;
% alternative from distance between spots, less robust than ratiometric
%           lambda = 0.45*(yright-yleft)

% update x0 from different photon count left/right
          y0 = (yleft+yright)/2;

% determination of z by astigmatism/defocus from second order moments
% proportionality constant (S-curve length) depends on distance
% between astigmatic focal lines, is now empirical
          Momxxright = sum(sum(XImage.^2.*dumright));
          Momyyright = sum(sum(YImage.^2.*dumright));
          Momxyright = sum(sum(XImage.*YImage.*dumright));
          Axxright = Momxxright-Nright*x0^2;
          Ayyright = Momyyright-Nright*yright^2;
          Axyright = Momxyright-Nright*x0*yright;
          Momxxleft = sum(sum(XImage.^2.*dumleft));
          Momyyleft = sum(sum(YImage.^2.*dumleft));
          Momxyleft = sum(sum(XImage.*YImage.*dumleft));
          Axxleft = Momxxleft-Nleft*x0^2;
          Ayyleft = Momyyleft-Nleft*yleft^2;
          Axyleft = Momxyleft-Nleft*x0*yleft;         
% astigmatic case
          scurveright = 2*Axyright/(Axxright+Ayyright);
          scurveleft = 2*Axyleft/(Axxleft+Ayyleft);
          scurve = scurveleft-scurveright;
%           scurve = 4*(Axyleft-Axyright)/(Axxleft+Ayyleft+Axxright+Ayyright);
          z0 = 400*scurve;
% defocus case
%           scurve = (Axxleft+Ayyleft-Axxright-Ayyright)/...
%             (Axxleft+Ayyleft+Axxright+Ayyright);
%           z0 = 1000*scurve;
          zmin = parameters.zspread(1);
          zmax = parameters.zspread(2);
          deltaz = zmax-zmin;
          z0 = max(z0,zmin-0.5*deltaz);
          z0 = min(z0,zmax+0.5*deltaz);
        case 'binary'
% estimation of lambda from distance between spots
%           ydiv = 0;
          ydiv = y0;
          dumright = dummat.*double(YImage>=ydiv);
          dumleft = dummat.*double(YImage<=ydiv);
          Nright = sum(sum(dumright));
          Nleft = sum(sum(dumleft));
          yright = sum(sum(YImage.*dumright))/Nright;
          yleft = sum(sum(YImage.*dumleft))/Nleft;
          lambda = 0.52*(yright-yleft);

% update x0 from different photon count left/right
          y0 = (yleft+yright)/2;

% determination of z by astigmatism/defocus from second order moments
% proportionality constant (S-curve length) depends on distance
% between astigmatic focal lines, is now empirical
          Momxxright = sum(sum(XImage.^2.*dumright));
          Momyyright = sum(sum(YImage.^2.*dumright));
          Momxyright = sum(sum(XImage.*YImage.*dumright));
          Axxright = Momxxright-Nright*x0^2;
          Ayyright = Momyyright-Nright*yright^2;
          Axyright = Momxyright-Nright*x0*yright;
          Momxxleft = sum(sum(XImage.^2.*dumleft));
          Momyyleft = sum(sum(YImage.^2.*dumleft));
          Momxyleft = sum(sum(XImage.*YImage.*dumleft));
          Axxleft = Momxxleft-Nleft*x0^2;
          Ayyleft = Momyyleft-Nleft*yleft^2;
          Axyleft = Momxyleft-Nleft*x0*yleft;         
% astigmatic case
          scurveright = 2*Axyright/(Axxright+Ayyright);
          scurveleft = 2*Axyleft/(Axxleft+Ayyleft);
          scurve = scurveleft-scurveright;
%           scurve = 4*(Axyleft-Axyright)/(Axxleft+Ayyleft+Axxright+Ayyright);
          z0 = 300*scurve;
% defocus case
%           scurve = (Axxleft+Ayyleft-Axxright-Ayyright)/...
%             (Axxleft+Ayyleft+Axxright+Ayyright);
%           z0 = 1000*scurve;
          zmin = parameters.zspread(1);
          zmax = parameters.zspread(2);
          deltaz = zmax-zmin;
          z0 = max(z0,zmin-0.5*deltaz);
          z0 = min(z0,zmax+0.5*deltaz);
        case 'sinusoidal'
% estimation of lambda
          
          Momxx = sum(sum(XImage.^2.*dummat));
          Axx = Momxx-Nph*x0^2;
          xwidth = sqrt(Axx/Nph);
          ydiv = xwidth;
          dumright = dummat.*double(YImage>=y0+ydiv);
          dumleft = dummat.*double(YImage<=y0-ydiv);
          Nright = sum(sum(dumright));
          Nleft = sum(sum(dumleft));
          yright = sum(sum(YImage.*dumright))/Nright;
          yleft = sum(sum(YImage.*dumleft))/Nleft;
          Ncenter = Nph-Nleft-Nright;
          lambda = parameters.lambdacentral+200*(Ncenter-Nright-Nleft)/Nph;
% alternative based on distance between satellite spots, may want to incorporate
% estimator from xylambda 3spots
          lambda = 0.3*(yright-yleft);
% determination of z by astigmatism/defocus from second order moments
% proportionality constant (S-curve length) depends on distance
% between astigmatic focal lines, is now empirical
          
          Momxxright = sum(sum(XImage.^2.*dumright));
          Momyyright = sum(sum(YImage.^2.*dumright));
          Momxyright = sum(sum(XImage.*YImage.*dumright));
          Axxright = Momxxright-Nright*x0^2;
          Ayyright = Momyyright-Nright*yright^2;
          Axyright = Momxyright-Nright*x0*yright;
          Momxxleft = sum(sum(XImage.^2.*dumleft));
          Momyyleft = sum(sum(YImage.^2.*dumleft));
          Momxyleft = sum(sum(XImage.*YImage.*dumleft));
          Axxleft = Momxxleft-Nleft*x0^2;
          Ayyleft = Momyyleft-Nleft*yleft^2;
          Axyleft = Momxyleft-Nleft*x0*yleft;         
% astigmatic case
          scurveright = 2*Axyright/(Axxright+Ayyright);
          scurveleft = 2*Axyleft/(Axxleft+Ayyleft);
          scurve = scurveleft-scurveright;
          z0 = 350*scurve;
          
        case 'azimuthramp'
% the appearing numerical constants depend on the number of zones
% and scaling of the azimuthal phase ramp
          mask = ones(Mx,My);
          if (My>Mx)
            mask(:,1:round((My-Mx)/2)) = 0;
            mask(:,My-round((My-Mx)/2)+1:My) = 0;
          end
          Momxx = sum(sum(XImage.^2.*dummat.*mask));
          Momyy = sum(sum(YImage.^2.*dummat.*mask));
          Momxy = sum(sum(XImage.*YImage.*dummat.*mask));
          Nphr = sum(sum(dummat.*mask));
          Axx = Momxx-Nphr*x0^2;
          Ayy = Momyy-Nphr*y0^2;
          Axy = Momxy-Nphr*x0*y0;
          helixangle = atan2(2*Axy,Ayy-Axx);
          z0 = 250*helixangle;
          zmin = parameters.zspread(1);
          zmax = parameters.zspread(2);
          deltaz = zmax-zmin;
          z0 = max(z0,zmin-0.5*deltaz);
          z0 = min(z0,zmax+0.5*deltaz);
          lambda = parameters.lambdacentral;
      end
      case 'aberrations'
        zmin = parameters.zrange(1);
        zmax = parameters.zrange(2);
        ImageSizez = (zmax-zmin)/2;
        DzImage = 2*ImageSizez/Mz; 
        ZImage = zmin+DzImage/2:DzImage:zmax;
        [~, zindex] = max(sum(sum(allspots(:,:,:,jcfg))));
%         z0 = ZImage(zindex);
        z0 = 0;
      case 'aberrationsamp'
        zmin = parameters.zrange(1);
        zmax = parameters.zrange(2);
        ImageSizez = (zmax-zmin)/2;
        DzImage = 2*ImageSizez/Mz; 
        ZImage = zmin+DzImage/2:DzImage:zmax;
        [~, zindex] = max(sum(sum(allspots(:,:,:,jcfg))));
%         z0 = ZImage(zindex);
        z0 = 0;
  end
  
% store estimated parameters depending on fit model. 
  switch parameters.fitmodel
    case 'xy'
      thetainit(1,jcfg) = x0;
      thetainit(2,jcfg) = y0;
      thetainit(3,jcfg) = Nph;
      thetainit(4,jcfg) = bg;
    case 'xyz'
      thetainit(1,jcfg) = x0;
      thetainit(2,jcfg) = y0;
      thetainit(3,jcfg) = z0;
      thetainit(4,jcfg) = Nph;
      thetainit(5,jcfg) = bg;
    case 'xylambda'
      thetainit(1,jcfg) = x0;
      thetainit(2,jcfg) = y0;
      thetainit(3,jcfg) = lambda;
      thetainit(4,jcfg) = Nph;
      thetainit(5,jcfg) = bg;
    case 'xyzlambda'
      thetainit(1,jcfg) = x0;
      thetainit(2,jcfg) = y0;
      thetainit(3,jcfg) = z0;
      thetainit(4,jcfg) = lambda;
      thetainit(5,jcfg) = Nph;
      thetainit(6,jcfg) = bg;
    case 'aberrations'
      thetainit(1,jcfg) = x0;
      thetainit(2,jcfg) = y0;
      thetainit(3,jcfg) = z0;
      thetainit(4:numparams-2,jcfg) = 0;
      thetainit(numparams-1,jcfg) = Nph;
      thetainit(numparams,jcfg) = bg;
    case 'aberrationsamp'
      thetainit(1,jcfg) = x0;
      thetainit(2,jcfg) = y0;
      thetainit(3,jcfg) = z0;
      thetainit(4:numparams-2,jcfg) = 0;
      thetainit(numparams-1,jcfg) = Nph;
      thetainit(numparams,jcfg) = bg;
  end
end
