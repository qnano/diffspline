function thetainit = initxyz(allspots,XImage,YImage,parameters)

dims = size(allspots);
if length(dims)>2
    [Mx,My,Mz] = size(allspots);
else
    [Mx,My] = size(allspots);
end


if length(dims)>2
  dummat = allspots(:,:,round((Mz+1)/2));
else
  dummat = allspots;
end

rimpixels = zeros(2*Mx+2*My-4,1);
rimpixels(1:Mx-1) = dummat(1:Mx-1,1);
rimpixels(Mx:2*Mx-2) = dummat(2:Mx,My);
rimpixels(2*Mx-1:2*Mx+My-3) = dummat(Mx,1:My-1);
rimpixels(2*Mx+My-2:2*Mx+2*My-4) = dummat(1,2:My);
bg = median(rimpixels);
bg = max(bg,1);

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

if (parameters.fitmodel=="aberrationsamp")
    thetainit=zeros(parameters.numparams,1);
    jcfg=1;
    thetainit(4:4+parameters.numparams_phaseaberr-1,jcfg)=parameters.aberrations(4:end,3);
    thetainit(4+parameters.numparams_phaseaberr:4+parameters.numparams_phaseaberr+parameters.numparams_ampaberr-1,jcfg)=parameters.aberrations(1:end,3);
    thetainit(1,jcfg) = 0;
    thetainit(2,jcfg) = 0;
    thetainit(3,jcfg) = 0;
    thetainit(end-1,jcfg) = Nph;
    thetainit(end,jcfg) = bg;
else
    jcfg=1;
    %thetainit(1,jcfg) = x0;
    %thetainit(2,jcfg) = y0;
    %thetainit(3,jcfg) = z0;
    thetainit(1,jcfg) = 0;
    thetainit(2,jcfg) = 0;
    thetainit(3,jcfg) = 250;
    thetainit(4,jcfg) = Nph;
    thetainit(5,jcfg) = bg;
    end
end