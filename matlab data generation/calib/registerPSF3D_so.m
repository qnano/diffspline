function [imout,shiftedstackn,shift,indgood]=registerPSF3D_so(imin,p,axs,filenumber)
if nargin<3
    axs={};
end
% perform correlation on:
% p.xrange
% p.yrange
% p.framerange
% can do 2D if length(p.framerange)=1
%cutout small volumes
if ~isfield(p,'xrange')
    p.xrange=1:size(imin,1);
end
if ~isfield(p,'yrange')
    p.yrange=1:size(imin,2);
end
if ~isfield(p,'framerange')
    p.framerange=1:size(imin,3);
end



numbeads=size(imin,4);
if numbeads==1
    imout=imin;
    shiftedstackn=imin;
    shift=[0 0 0 ];
    indgood=true;
    return
end
if ~isempty(p.zshiftf0)
    zshiftf0=p.zshiftf0;
else
    zshiftf0=zeros(numbeads,1);
end
imina=imin;
numref=max(round(size(imina,4)*.5),min(5,size(imina,4))); %align to average of several beads; size/2<=numref<=5
numref=1; %align to a sigle bead only;
avim=nanmean(imina(:,:,:,p.sortind(1:numref)),4);
% avim=nanmean(smallim,4);
ph=p;
lcc=ceil((min(13,length(p.yrange))-1)/2);
mp=ceil(((length(p.yrange))-1)/2)+1;
ph.yrange=p.yrange(mp-lcc:mp+lcc);
ph.xrange=p.xrange(mp-lcc:mp+lcc);

%new algorithm to try:
%1. align with all frames
ph.framerange=1:size(avim,3);
[shiftedstack,shift,cc]=aligntoref(avim,imina, zshiftf0,ph);

%calculate good ones, 
shiftedstackn=normalizstack(shiftedstack,p);
indgood=true(1,size(shiftedstackn,4));
[indgood]=getoverlap(shiftedstackn,shift,ph,indgood);
meanim=nanmean(shiftedstack(:,:,:,indgood),4);meanim(isnan(meanim))=avim(isnan(meanim));   

%do central correlation using shiftedstack
ph.framerange=p.framerange;
[shiftedstack,shift2,cc]=aligntoref(meanim,shiftedstack, 0*zshiftf0,ph);
shift=shift+shift2;

shiftedstackn=normalizstack(shiftedstack,p);

indgood=true(1,size(shiftedstackn,4));
[indgood]=getoverlap(shiftedstackn,shift,ph,indgood);
[indgood]=getoverlap(shiftedstackn,shift,ph,indgood);
[indgood,res,normglobal,co,cc2]=getoverlap(shiftedstackn,shift,ph,indgood);
shiftedstackn=shiftedstackn/normglobal;

%save('./logs/indgood.mat', "indgood", "shiftedstackn");
imout=shiftedstackn(:,:,:,:);%shiftedstackn(:,:,:,indgood); %nanmean(shiftedstackn(:,:,:,indgood),4);
shiftedstackn(1,end,:,~indgood)=nanmax(shiftedstackn(:));
shiftedstackn(1,:,1,~indgood)=nanmax(shiftedstackn(:));


% smallim=zeros(length(p.xrange),length(p.yrange),length(p.framerange),size(imin,4));
% 
% if ~isempty(p.zshiftf0)
%     zshiftf0=p.zshiftf0;
% else
%     zshiftf0=zeros(numbeads,1);
% end
% 
% for k=1:size(imin,4)
%     frh=round(p.framerange-zshiftf0(k)); %makes sure, all beads are at same z position
%     try
%     smallim(:,:,:,k)=imin(p.xrange,p.yrange,frh,k);
%     catch err %range out 
%     end
% end
% avim=nanmean(imin,4);
% 
% xn=1:size(imin,1);yn=1:size(imin,2);zn=1:size(imin,3);
% [Xq,Yq,Zq]=meshgrid(yn,xn,zn);

% xns=1:size(smallim,1);yns=1:size(smallim,2);zns=1:size(smallim,3);
% [Xqs,Yqs,Zqs]=meshgrid(yns,xns,zns);

% meanim=[];
% refim=avim(p.xrange,p.yrange,p.framerange);
% 
% simin=size(imin);
% shiftedstack=zeros(simin(1),simin(2),simin(3),numbeads)+NaN;
% 
% for k=1:numbeads
%     goodframes=squeeze(nansum(nansum(smallim(:,:,:,k),1),2))>0;
%     if p.alignz
%         [shift(k,:),cc(k)]=get3Dcorrshift(refim(:,:,goodframes),smallim(:,:,goodframes,k));
%     else
%         if any(goodframes)
% %             smallimshiftf0=interp3(smallim(:,:,:,k),Xqs,Yqs,Zqs-double(zshiftf0(k)),'cubic',0);
%             [shift(k,:),cc(k)]=get2Dcorrshift(refim(:,:,goodframes),smallim(:,:,goodframes,k));
% %             [shift(k,:),cc(k)]=get2Dcorrshift(refim(:,:,goodframes),smallimshiftf0);
%         else
%             shift(k,:)=[0 0 0];cc(k)=NaN;
%         end
%     end
% %     
%     shiftedh=interp3(imin(:,:,:,k),Xq-shift(k,2),Yq-shift(k,1),Zq-shift(k,3)-double(zshiftf0(k)),'cubic',0);
%     shiftedstack(:,:,:,k)=shiftedh;
%     meanim=nanmean(shiftedstack(:,:,:,1:k),4);
%     meanim(isnan(meanim))=avim(isnan(meanim));
%     
%     refim=meanim(p.xrange,p.yrange,p.framerange);
% end

% shiftedstackn=normalizstack(shiftedstack,p);
% 
% indgood=true(1,size(shiftedstackn,4));
% [indgood]=getoverlap(shiftedstackn,shift,p,indgood);
% [indgood]=getoverlap(shiftedstackn,shift,p,indgood);
% [indgood,res,normglobal,co,cc2]=getoverlap(shiftedstackn,shift,p,indgood);
% shiftedstackn=shiftedstackn/normglobal;
% 
% imout=nanmean(shiftedstackn(:,:,:,indgood),4);
% shiftedstackn(1,end,:,~indgood)=nanmax(shiftedstackn(:));
% shiftedstackn(1,:,1,~indgood)=nanmax(shiftedstackn(:));

col=lines(max(filenumber));
if length(axs)>0
    leg={};
    hold(axs{1},'off')
    for k=1:max(filenumber)
        fh=filenumber==k;
        if any((~indgood)&fh)
            plot(axs{1},(res((~indgood)&fh)),cc2((~indgood)&fh),'x','Color',col(k,:));
            hold(axs{1},'on')
            leg{end+1}=['x' num2str(k) ':' num2str(sum((~indgood)&fh))];
        end
        if any(indgood&fh) 
            plot(axs{1},(res(indgood&fh)),cc2(indgood&fh),'*','Color',col(k,:));
            hold(axs{1},'on')
            leg{end+1}=[num2str(k) ':' num2str(sum((indgood)&fh))];
        end
    end
    legend(leg);
    xlabel(axs{1},'residulas')
    ylabel(axs{1},'cross-correlation value')
    drawnow
end

if length(axs)>1
    imageslicer(vertcat(avim,imout),'Parent',axs{2}.Parent)
end

end

function [indgood,res,normamp,co,cc]=getoverlap(shiftedstackn,shift,p,indgood)
refimn=nanmean(shiftedstackn(p.xrange,p.yrange,p.framerange,indgood),4);
for k=size(shiftedstackn,4):-1:1
    imh=shiftedstackn(p.xrange,p.yrange,p.framerange,k);
    badind=isnan(imh)|isnan(refimn);
    cc(k)=sum(refimn(~badind).*imh(~badind))/(sum(refimn(~badind))*sum(imh(~badind)))*sum(~badind(:));  
end

normamp=nanmax(refimn(:));
shiftedstackn=shiftedstackn/normamp;
refimn=refimn/normamp;
for k=size(shiftedstackn,4):-1:1
     sim=shiftedstackn(p.xrange(2:end-1),p.yrange(2:end-1),p.framerange,k);
     dv=(refimn(2:end-1,2:end-1,:)-sim).^2;
    res(k)=sqrt(nanmean(dv(:)));
end
rescc=res./cc;
rescc(abs(shift(:,1))>3|abs(shift(:,2))>3)=NaN;
[a,b]=robustMean(rescc(cc>0));
if isnan(b)
    a=nanmean(rescc);b=nanstd(rescc);
end
co=a+2.5.*b;
indgood=rescc<=co;
end

function out=normalizstack(in,p)
sin=size(in);
  midp=round((length(p.xrange)+1)/2);
    xr=p.xrange(midp-3:midp+3);yr=p.yrange(midp-3:midp+3);
if p.beadfilterf0 
    out=0*in+NaN;
  
    for k=1:sin(4)
        imh=in(xr,yr,p.framerange,k);
        nm=nanmean(imh(:));
        if nm>0
        out(:,:,:,k)=in(:,:,:,k)/nm;
        end
    end
else %use fitting
    inh=in;
    out=0*in+NaN;
    for iter=1:4
        meanim=nanmean(inh,4);
        for k=1:sin(4)
            imh=inh(:,:,:,k);
            if all(isnan(imh))
                continue
            end
            ims=imh(xr,yr,p.framerange);
            meanims=meanim(xr,yr,p.framerange);
            isn=isnan(ims)|isnan(meanims);
            intcutoff=meanims>quantile(meanims(:),0.75);
            indg=~isn&intcutoff;
            ratio=ims(indg)./meanims(indg);
            
            factor=nanmedian(ratio(:));
            if factor>0
                out(:,:,:,k)=imh/factor;
            end
        end
        inh=out;
    end

end

end

function [shiftedstack,shift,cc]=aligntoref(avim,imina, zshiftf0,p)
xn=1:size(imina,1);yn=1:size(imina,2);zn=1:size(imina,3);
smallim=zeros(length(p.xrange),length(p.yrange),length(p.framerange),size(imina,4));
for k=1:size(imina,4)
    frh=round(p.framerange-zshiftf0(k)); %makes sure, all beads are at same z position
    try
    smallim(:,:,:,k)=imina(p.xrange,p.yrange,frh,k);
    catch err %range out 
    end
end
[Xq,Yq,Zq]=meshgrid(yn,xn,zn);
refim=avim(p.xrange,p.yrange,p.framerange);
numbeads=size(imina,4);
simin=size(imina);
shiftedstack=zeros(simin(1),simin(2),simin(3),numbeads)+NaN;

for k=1:numbeads
        p.status.String=['calculate shift of individual PSFs: ' num2str(k) ' of ' num2str(numbeads)]; drawnow
    goodframes=squeeze(nansum(nansum(smallim(:,:,:,k),1),2))>0;
    if p.alignz
        [shift(k,:),cc(k)]=get3Dcorrshift(refim(:,:,goodframes),smallim(:,:,goodframes,k));
    else
        if any(goodframes)
            [shift(k,:),cc(k)]=get2Dcorrshift(refim(:,:,goodframes),smallim(:,:,goodframes,k));
        else
            shift(k,:)=[0 0 0];cc(k)=NaN;
        end
    end

    shiftedh=interp3(imina(:,:,:,k),Xq-shift(k,2),Yq-shift(k,1),Zq-shift(k,3)-double(zshiftf0(k)),'cubic',0);
    shiftedstack(:,:,:,k)=shiftedh;
end
end

