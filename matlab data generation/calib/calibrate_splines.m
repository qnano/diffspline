function [splinefit, p]=calibrate_splines(samedepthPSFs,splinefit, p, midrange) %, z0reference)
        corrPSF = nanmean(samedepthPSFs,4);

        %cut out the central part of the PSF correspoinding to the set
        %Roisize in x,y and z
        scorrPSF=size(corrPSF);
        
        x=round((scorrPSF(1)+1)/2);y=round((scorrPSF(2)+1)/2);

        dRx=round((p.ROIxy-1)/2);
        if ~isfield(p,'ROIz') || isnan(p.ROIz)
            p.ROIz=size(corrPSF,3);
        end
            dzroi=round((p.ROIz-1)/2);
        
        rangex=x-dRx:x+dRx;
        rangey=y-dRx:y+dRx;

        z=midrange;%always same reference: z=f0
        rangez=max(1,z-dzroi):min(size(corrPSF,3),z+dzroi);
        z0reference=find(rangez>=z,1,'first');
        
        %normalize PSF
        centpsf=corrPSF(rangex,rangey,z-1:z+1); %cut out rim from shift
%         centpsf=corrPSF(2:end-1,2:end-1,2:end-1); %cut out rim from shift
        minPSF=nanmin(centpsf(:)); %nanmin(centpsf,[],[1,2,3]);
        corrPSFn=corrPSF-minPSF;
%         corrPSFn=corrPSF;
        intglobal=nanmean(nansum(nansum(corrPSFn(rangex,rangey,z-1:z+1),1),2));
        corrPSFn=corrPSFn/intglobal;

        corrPSFn(isnan(corrPSFn))=0;
        corrPSFn(corrPSFn<0)=0;
        corrPSFs=corrPSFn(rangex,rangey,rangez,:);


        %calculate effective smoothing factor. For dz=10 nm, pixelsize= 130
        %nm, a value around 1 produces visible but not too much smoothing.
        lambdax=p.smoothxy/p.cam_pixelsize_um(1)/100000;
        lambdaz=p.smoothz/p.dz*100;
        lambda=[lambdax lambdax lambdaz];
        %calculate smoothed bsplines
        b3_0=bsarray(double(corrPSFs),'lambda',lambda);

        %calculate smoothed volume
        zhd=1:1:b3_0.dataSize(3);
        dxxhd=1;
        [XX,YY,ZZ]=meshgrid(1:dxxhd:b3_0.dataSize(1),1:dxxhd:b3_0.dataSize(2),zhd);
        p.status.String='calculating cspline coefficients in progress';drawnow
        corrPSFhd = interp3_0(b3_0,XX,YY,ZZ,0);
        
        %calculate cspline coefficients
%         spline = Spline3D_v2(corrPSFhd);
%         coeff = spline.coeff;
        coeff = Spline3D_interp(corrPSFhd);
       
        %assemble output structure for saving
        bspline.bslpine=b3_0;
        cspline.coeff=coeff;
        cspline.z0=z0reference;%round((b3_0.dataSize(3)+1)/2);
        cspline.dz=p.dz;
        cspline.x0=round((p.ROIxy-1)/2)+1;
        bspline.z0=round((b3_0.dataSize(3)+1)/2);
        bspline.dz=p.dz;            
        splinefit.bspline=bspline;
        p.z0=cspline.z0;
        
        splinefit.PSFaverage=corrPSFs;
        
        splinefit.PSFsmooth=corrPSFhd;
        splinefit.cspline=cspline;

%         splinefit.allstacks=allstacks;
%         splinefit.shiftedstack=shiftedstack;
%         splinefit.shift=shift;
        
%         PSFgood=false;
%         %plot graphs
%         if PSFgood       
%             ax=axes(uitab(p.tabgroup,'Title','PSFz'));
%              framerange0=max(p.fminmax(1)):min(p.fminmax(2));
%              halfroisizebig=(size(shiftedstack,1)-1)/2;         
%             ftest=z;
%             xt=x;
%             yt=y;
%             zpall=squeeze(shiftedstack(xt,yt,:,beadgood));
%             zpall2=squeeze(allrois(xt,yt,:,beadgood));
%             xpall=squeeze(shiftedstack(:,yt,ftest,beadgood));
%             xpall2=squeeze(allrois(:,yt,ftest,beadgood));
%             for k=1:size(zpall,2)
%                 zpall2(:,k)=zpall2(:,k)/nanmax(zpall2(:,k));
%                 xpall2(:,k)=xpall2(:,k)/nanmax(xpall2(:,k));                
%             end           
%             zprofile=squeeze(corrPSFn(xt,yt,:));
% %             mphd=round((size(corrPSFhd,1)+1)/2);
%                  
%             xprofile=squeeze(corrPSFn(:,yt,ftest));
%             mpzhd=round((size(corrPSFhd,3)+1)/2+1);
%             dzzz=round((size(corrPSFn,3)-1)/2+1)-mpzhd;
%             dxxx=0.1;
%             xxx=1:dxxx:b3_0.dataSize(1);
% %             zzzt=0*xxx+mpzhd+dzzz-1;
%             zzzt=0*xxx+ftest;
%             xbs= interp3_0(b3_0,0*xxx+b3_0.dataSize(1)/2+.5,xxx,zzzt);
%             zzz=1:dxxx:b3_0.dataSize(3);xxxt=0*zzz+b3_0.dataSize(1)/2+.5;
%             zbs= interp3_0(b3_0,xxxt,xxxt,zzz); 
%             hold(ax,'off')
%              h1=plot(ax,framerange0,zpall(1:length(framerange0),:),'c');
%              hold(ax,'on')
%             h2=plot(ax,framerange0',zprofile(1:length(framerange0)),'k*');
%             h3=plot(ax,zzz+rangez(1)+framerange0(1)-2,zbs,'b','LineWidth',2);
%             xlabel(ax,'frames')
%             ylabel(ax,'normalized intensity')
%             ax.XLim(2)=max(framerange0);ax.XLim(1)=min(framerange0);
%             title(ax,'Profile along z for x=0, y=0');
%             
%             legend([h1(1),h2,h3],'individual PSFs','average PSF','smoothed spline')
%             
%             xrange=-halfroisizebig:halfroisizebig;
%              ax=axes(uitab(p.tabgroup,'Title','PSFx'));
%             hold(ax,'off')
%             h1=plot(ax,xrange,xpall,'c');
%             hold(ax,'on')
%             h2=plot(ax,xrange,xprofile,'k*');
%             h3=plot(ax,(xxx-(b3_0.dataSize(1)+1)/2),xbs,'b','LineWidth',2);
%             xlabel(ax,'x (pixel)')
%             ylabel(ax,'normalized intensity')
%             title(ax,'Profile along x for y=0, z=0');
%              legend([h1(1),h2,h3],'individual PSFs','average PSF','smoothed spline')
%             
%             drawnow
%             
%             %quality control: refit all beads
%             if isempty(stackcal_testfit)||stackcal_testfit
%                 ax=axes(uitab(p.tabgroup,'Title','validate'));
%                 testallrois=allrois(:,:,:,beadgood);
%                 testallrois(isnan(testallrois))=0;
%                 zall=testfit(testallrois,cspline.coeff,p,{},ax);
%                 corrPSFfit=corrPSF/max(corrPSF(:))*max(testallrois(:)); %bring back to some reasonable photon numbers;
%                 zref=testfit(corrPSFfit,cspline.coeff,p,{'k','LineWidth',2},ax);
%                 drawnow
%             end
%         end 