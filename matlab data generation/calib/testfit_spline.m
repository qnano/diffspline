function zs=testfit(teststack,coeff,shiftxy, p,linepar,ax)
if nargin<4
    linepar={};
elseif ~iscell(linepar)
    linepar={linepar};
end
roifit=17;
d=round((size(teststack,1)-roifit)/2);
            range=d+1:d+roifit;

numstack=size(teststack,4);
t=tic;
% f=figure(989);ax2=gca;hold off
dx1=[];
dx2=[];dy1=[];
dy2=[];

f=figure(134);ax2=gca;hold(ax2,'off')
    for k=1:size(teststack,4)
        if toc(t)>1
            p.status.String=['fitting test stacks: ' num2str(k/numstack,'%1.2f')];drawnow
            t=tic;
        end
        if contains(p.modality,'2D')
            fitmode=6;
        else
            fitmode=5;
        end
        fitstack=single(squeeze(teststack(range,range,:,k,:)));
        coeffh(:,:,:,:,1)=single(coeff{1});
        coeffh(:,:,:,:,2)=single(coeff{2});
        shared=[1 1 1 0 0]';
        nfits=size(fitstack,3);
        npar=5;
        dT=zeros(npar,2,nfits);
        dT(1,2,:)=shiftxy(k,2);
        dT(2,2,:)=shiftxy(k,1);
        iterations=150;
%         [PM,CRLBM, LLM,update, error] =  kernel_MLEfit_Spline_LM_multichannel_finalized(fitstack,coeffh, shared,dT,50);
        sharedA = repmat(shared,[1 size(fitstack,3)]);
        try
        [P,CRLB, LL] =GPUmleFit_LM_MultiChannel(fitstack,int32(sharedA),iterations,coeffh,single(dT));
        catch err
              [P,CRLB, LL] =CPUmleFit_LM_MultiChannel(fitstack,int32(sharedA),iterations,coeffh,single(dT));
        end
        
        
        [P,CRLB, LL,residuals] =CPUmleFit_LM_MultiChannel_R(fitstack,int32(sharedA),iterations,coeffh,single(dT));
        %compare with sinlge fits
%         [P1,CRLB1, LL1] =mleFit_LM(fitstack(:,:,:,1),5,iterations,single(coeff{1}),0,1);
%         [P2,CRLB2, LL2] =mleFit_LM(fitstack(:,:,:,2),5,iterations,single(coeff{2}),0,1);
%         %compare with all free fits
%         [Pf,CRLBr, LLf] =CPUmleFit_LM_MultiChannel(fitstack,int32(sharedA*0),iterations,coeffh,single(dT));
%         
%         dx1(k)=median(P1(:,1));
%         dy1(k)=median(P1(:,2));
%          dx2(k)=median(P2(:,1));
%         dy2(k)=median(P2(:,2));
        %define one as reference and plot differences 
%         figure(78);
%         rs=01;
%         subplot(1,2,1)
% hold off
%         plot(P1(:,1),P2(:,1)-squeeze(dT(1,2,:)),'.')
% hold on
%         plot(Pf(:,1),Pf(:,2)-0*shiftxy(k,2),'.')
%         plot(P1(:,1),P(:,1),'.')
%         plot([-1 1]*rs+(roifit-1)/2,[-1 1]*rs+(roifit-1)/2)
%          title(dT(1,2))
%         xlabel('x1 single fit')
%         ylabel('x ')
%         legend('x2 individual fit','x2 not linked','x linked')
%  
%                 subplot(1,2,2)
% hold off
%         plot(P1(:,2),P2(:,2)-squeeze(dT(2,2,:)),'.')
% hold on
%         plot(Pf(:,3),Pf(:,4)-0*shiftxy(k,1),'.')
%         plot(P1(:,2),P(:,2),'.')
%         plot([-1 1]*rs+(roifit-1)/2,[-1 1]*rs+(roifit-1)/2)
%         title(dT(2,2))
%         xlabel('y1 single fit')
%         ylabel('y ')
%         legend('y2 individual fit','y2 not linked','y linked')
%       
%         [P] =  mleFit_LM(single(squeeze(teststack(range,range,:,k))),fitmode,100,single(coeff),0,1);
        
        z=(1:size(P,1))'-1;

        znm=(P(:,3)-p.z0)*p.dz;
        plot(ax,z,znm,linepar{:})
        hold(ax,'on')
        xlabel(ax,'frame')
        ylabel(ax,'zfit (nm)')
        zs(:,k)=P(:,3);
%         

plot(ax2,P(:,1),P(:,2),'.')
hold(ax2,'on')


%         figure(104);
%         plot(z,P1(:,5)-z,z,P2(:,5)-z,z,Pf(:,5)-z,z,Pf(:,6)-z,z,P(:,3)-z,'k')
%         legend('z1','z2','zi1','zi2','zg');
%         xlabel(ax,'frame')
%         ylabel(ax,'zfit (nm)')
%         ylim([-5 5])
        
        if 0% imageslicer to test
%             coord=P1(:,[1 2 5 3 4]);
            coord=P(:,[1 2 3 4 6]);
            coord2=coord;
            coord2(:,1)=coord2(:,1)+squeeze(dT(1,2,:));
            coord2(:,2)=coord2(:,2)+squeeze(dT(2,2,:));
            img1=renderPSF(coeff{1},coord,size(fitstack,1));
            img2=renderPSF(coeff{2},coord2,size(fitstack,1));
            imall=[fitstack(:,:,:,1),img1; fitstack(:,:,:,2),img2];
            res=[fitstack(:,:,:,1)-img1, 0*img1;fitstack(:,:,:,2)-img2,0*img2];
            ims(:,:,:,1)=imall;ims(:,:,:,2)=res;
            f=figure(105);
            imageslicer(ims,'Parent',f);
        end
        
% test for the returned photons and photons in the raw image        
%         phot=P(:,3); bg=P(:,4);
%         totsum=squeeze(nansum( nansum(teststack(range,range,:,k),1),2));
%         totsum=totsum-squeeze(min(min(teststack(range,range,:,k),[],1),[],2))*length(range)^2;
%         photsum=phot+0*bg*length(range)^2;
%         plot(ax2,z,(photsum-totsum)./totsum,'.')
%         hold(ax2,'on')
    end
    
%     figure(222)
%     zzz=1:length(dx1);
%     plot(zzz,dx1-dx2,'x',zzz,dy1-dy2,'o',zzz,-shiftxy(:,1),zzz,-shiftxy(:,2))
    
end
