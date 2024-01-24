%  Copyright (c)2017 Ries Lab, European Molecular Biology Laboratory,
%  Heidelberg.
%  
%  This file is part of GPUmleFit_LM Fitter.
%  
%  GPUmleFit_LM Fitter is free software: you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation, either version 3 of the License, or
%  (at your option) any later version.
%  
%  GPUmleFit_LM Fitter is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%  
%  You should have received a copy of the GNU General Public License
%  along with GPUmleFit_LM Fitter.  If not, see <http://www.gnu.org/licenses/>.
%  
%  
%  Additional permission under GNU GPL version 3 section 7

%%
function [PSFsmooth,PSFaverage,zstack,shiftedzstack,shifts]=calibrate3D(p)
% p.filelist
% p.outputfile
% p.dz
% p.modality
% p.zcorr
% p.ROIxy
% p.ROIz
% p.smoothxy
% p.smoothz
% p.gaussrange
% p.filter;
% p.zcorrframes
% p.gaussroi
% p.fov =[x1 y1 x2 y2] 
% p.mindistance
% p.emgain
% p.xrange
% p.yrange
% p.positions (get beads only here)
% p.smap (called from SMAP: extended functionality)
if ~isfield(p,'smap')
    p.smap=false;
    imageRoi=zeros(2,1); 
else
    imageRoi=p.imageRoi; %position of image on Chip. This is important, if the calibration file used a different ROI than the actual measurement
end
if ~isfield(p,'xrange')
    p.xrange=[-inf inf]; %p.yrange=[-inf inf]; 
end

if ~isfield(p,'yrange')
    p.yrange=[-inf inf]; 
end

if ~isfield(p,'emgain')
    p.emgain=0;
end

if ~isfield(p,'smoothxy')
    p.smoothxy=0;
end

%get bead positions
p.status.String='Load files and segment beads';drawnow
f=figure('Name','Bead calibration');
p.tabgroup=uitabgroup(f);
%get beads from images
[beads,p]=images2beads_so(p);

%get positions of beads
for k=length(beads):-1:1
    beadposx(k)=beads(k).pos(1);
    beadposy(k)=beads(k).pos(2);
end

%if only to take beads in a certain range, remove others
if isfield(p,'fov')&&~isempty(p.fov)
    indbad=beadposx<p.fov(1)| beadposx>p.fov(3)|beadposy<p.fov(2)|beadposy>p.fov(4);
    beads=beads(~indbad);
end


%remove beads that are closer together than mindistance
if isfield(p,'mindistance')&&~isempty(p.mindistance)
    indgood=true(length(beads),1);
    for k=1:length(beads)
        for l=k+1:length(beads)
            if beads(k).filenumber == beads(l).filenumber && sum((beads(k).pos-beads(l).pos).^2)<p.mindistance^2
                indgood(k)=false;
                indgood(l)=false;
            end
        end
    end 
    beads=beads(indgood); 
end  
if isempty(beads)
    warndlg('Could not find and segment any bead. ROI size too large?')
    p.status.String='error: could not find and segment any bead...';
    return
end

p.midpoint=round(size(beads(1).stack.image,3)/2); %reference for beads  %% midpoint == center in z (41 out of 81 planes)
p.ploton=false;

% beads=beads([2 3])
% beads(2)=beads(1); 
% beads(3:end)=[];

if contains(p.modality,'astig') % || contains(p.modality,'2D') 2D treated now as arbitrary
    %determine sx,sy
%     disp('fit beads to get sx,sy')
    t=tic;
    p.status.String=['Gaussian fit of beads to get spatial parameters '];drawnow
    for k=1:length(beads)
        stackh=single(beads(k).stack.image);
        s=size(stackh); 
        d=round((s(1)-p.gaussroi)/2);
        stack=stackh(d+1:end-d,d+1:end-d,:);
        %fit bead bead stacks with Gaussian model
%         if contains(p.modality,'astig')
            P=mleFit_LM(stack,4,100,1,0,1);
            beads(k).loc.PSFxpix=P(:,5);
            beads(k).loc.PSFypix=P(:,6);
            beads(k).loc.phot=P(:,3);
            beads(k).f0=stackas2z_so(beads(k).loc.PSFxpix,beads(k).loc.PSFypix,beads(k).loc.frames,beads(k).loc.phot,p);
%         else
%             P=mleFit_LM(stack,2,100,1,0,1);
%             beads(k).loc.PSFxpix=P(:,5);
%             beads(k).loc.PSFypix=P(:,5);
%             beads(k).loc.phot=P(:,3);
%             beads(k).f0=stackas2z2D_so(beads(k).loc.PSFxpix,beads(k).loc.frames,beads(k).loc.phot,p);
%         end
        
        beads(k).loc.bg=P(:,4);
        %determine true position of the beads as the position, where PSFx==PSFy
        
        ind=find(beads(k).loc.frames<=beads(k).f0,1,'last');
        if isnan(beads(k).f0)||isempty(ind)
            ind=1;
        end
        beads(k).psfx0=beads(k).loc.PSFxpix(ind);
        beads(k).psfy0=beads(k).loc.PSFypix(ind);
        if toc(t)>1
            p.status.String=['Gaussian fit of beads to get spatial parameters: ' num2str(k) ' of ' num2str(length(beads))];
            drawnow
            t=tic;
        end
    end
    %remove beads for which no position could be found
    badind=isnan([beads(:).f0]);
    beads(badind)=[];
else
    f0g=p.midpoint;
    for k=1:length(beads)
        beads(k).f0=f0g;
    end
end

%get positions of beads
for k=length(beads):-1:1
    beadposxs(k)=beads(k).pos(1);
    beadposys(k)=beads(k).pos(2);
    beadfilenumber(k)=beads(k).filenumber;
end

%spatially dependent calibration
tgmain=p.tabgroup;
for X=1:length(p.xrange)-1
    for Y=1:length(p.yrange)-1
        if length(p.xrange)>2||length(p.yrange)>2
            ht=uitab(tgmain,'Title',['X' num2str(X) 'Y' num2str(Y)]);
            p.tabgroup=uitabgroup(ht);
        end
        
        indgood=beadposxs+imageRoi(1)< p.xrange(X+1) & beadposxs+imageRoi(1)>p.xrange(X) & beadposys+imageRoi(2)<p.yrange(Y+1) & beadposys+imageRoi(2)>p.yrange(Y);
        beadsh=beads(indgood);
        
        for k=1:max(beadfilenumber)
            indfile=(beadfilenumber==k)&indgood;
            p.fileax(k).NextPlot='add';
            scatter(p.fileax(k),beadposxs(indfile),beadposys(indfile),60,[1 1 1])
            scatter(p.fileax(k),beadposxs(indfile),beadposys(indfile),50)
        end
        if isempty(beadsh)
            disp(['no beads found in part' num2str(p.xrange(X:X+1)) ', ' num2str(p.yrange(Y:Y+1))])
            continue
        end

        if contains(p.modality,'astig')
            %get calibration for Gaussian fit
            p.status.String='get spline approximation';drawnow
            p.ax_z=axes(uitab(p.tabgroup,'Title','sx(z), sy(z)'));
            [spline_curves,indgoodc,curves]=getspline_so(beadsh,p); 
            gausscal.spline_curves=spline_curves;
            drawnow
        else
            indgoodc=true(size(beadsh));
            gausscal=[];
            p.ax_z=[];
        end


        % get cspline calibration
        p.status.String='get cspline calibration';drawnow
        [csplinecal,indgoods,shift, midrange]=getstackcal_so(beadsh(indgoodc),p);
        for i=1:p.n_realizations:size(csplinecal.PSFs,4)
            samedepthPSFs=csplinecal.PSFs(:,:,:,i:i+p.n_realizations-1);
            [csplinecal_new,p]=calibrate_splines(samedepthPSFs,csplinecal,p, midrange);
            PSFaverage=csplinecal_new.PSFaverage;
            PSFsmooth=csplinecal_new.PSFsmooth;
            zstack=csplinecal_new.allstacks;
            shiftedzstack=csplinecal_new.shiftedstack;
            shifts=csplinecal_new.shift;
            cspline.coeff=single(csplinecal_new.cspline.coeff);
            cspline.dz=csplinecal_new.cspline.dz;
            cspline.z0=csplinecal_new.cspline.z0;
            cspline.x0=csplinecal_new.cspline.x0;

            gausscal=[];
            gauss_sx2_sy2=[];
            gauss_zfit=[];
            p.ax_sxsy=[];
            
            cspline_all=csplinecal_new;
            parameters=rmfield(p,{'tabgroup','status','ax_z','ax_sxsy','fileax'});
            save(strcat(p.outputfile, sprintf('%04d',p.depths(i)), '_3Dcorr.mat'),'gausscal','cspline_all','gauss_sx2_sy2','gauss_zfit','cspline','parameters');

        end
        



        %name=char(split(p.outputfile, '\'));
        %name=append('..\diff-spline-psf\', name(end,1:end-4),'.tif');        
        %t = Tiff(name, 'w'); 
        %tagstruct.ImageLength = size(csplinecal.PSFsmooth,1);
        %tagstruct.ImageWidth = size(csplinecal.PSFsmooth, 2);
        %tagstruct.Compression = Tiff.Compression.None; 
        %tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP; 
        %tagstruct.Photometric = Tiff.Photometric.MinIsBlack; 
        %tagstruct.BitsPerSample = 32;
        %tagstruct.SamplesPerPixel = 1;
        %tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; 
        %t.setTag(tagstruct); 
        %t.write(csplinecal.PSFsmooth); 
        %t.close();

        %imwrite(csplinecal.PSFsmooth(:,:,1), name, "tif")
        %for i=2:p.fminmax(2)
        %    imwrite(csplinecal.PSFsmooth(:,:,i), name, "WriteMode", "append")
        %end


        icf=find(indgoodc);
        icfs=icf(indgoods);
        

        if contains(p.modality,'astig')
            photbead=10^5; %corr PSF normalized to 1. As MLE is used, this screws up statistics totally. Thus assign bright signal to bead.
            stackb=csplinecal.PSF;
            stackb=(stackb)*photbead;
            mp=ceil(size(stackb,1)/2);dx=floor(p.gaussroi/2);

            stack=single(stackb(mp-dx:mp+dx,mp-dx:mp+dx,:));
            P=mleFit_LM(stack,4,200,1,0,1);
            ch.sx=double(P(:,5));
            ch.sy=double(P(:,6));
            f0m=median([beadsh(icfs).f0]);
            ch.z=double(((1:size(stack,3))'-f0m)*p.dz);

            p.ax_sxsy=axes(uitab(p.tabgroup,'Title','Gauss cal'));
            p.ax_z.NextPlot='add';
            p.status.String='get Gauss model calibration';drawnow
            gausscalh=getgausscal_so(ch,p); 
            legend(p.ax_z,'bad bead data','good bead data','spline fit sx','spline fit sy','average PSF','average PSF','Gauss zfit','Gauss zfit')

            gausscal=copyfields(gausscal,gausscalh);
            gauss_zfit=single(gausscal.fitzpar);
            gauss_sx2_sy2=gausscal.Sx2_Sy2;
        else
            gausscal=[];
            gauss_sx2_sy2=[];
            gauss_zfit=[];
            p.ax_sxsy=[];
            
        end
%         cspline_all=csplinecal;
%         PSF=csplinecal.PSF;
%         SXY(X,Y)=struct('gausscal',gausscal,'cspline_all',cspline_all,'gauss_sx2_sy2',gauss_sx2_sy2,'gauss_zfit',gauss_zfit,...
%             'cspline',cspline,'Xrangeall',p.xrange+imageRoi(1),'Yrangeall',p.yrange+imageRoi(2),'Xrange',p.xrange([X X+1])+imageRoi(1),...
%             'Yrange',p.yrange([Y Y+1])+imageRoi(2),'posind',[X,Y],'EMon',p.emgain,'PSF',PSF);
    end
end
    
% parameters=rmfield(p,{'tabgroup','status','ax_z','ax_sxsy','fileax'});
%     
% p.status.String='save calibration';drawnow
% if p.smap
%     save(p.outputfile,'SXY','parameters');
% else
%     save(p.outputfile,'gausscal','cspline_all','gauss_sx2_sy2','gauss_zfit','cspline','parameters');
% end
p.status.String='Calibration done';drawnow
end




