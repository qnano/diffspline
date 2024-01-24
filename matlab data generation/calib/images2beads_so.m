function [b,p]=images2beads_so(p)
% addpath('bfmatlab')
fs=p.filtersize;
h=fspecial('gaussian',2*round(fs*3/2)+1,fs);
fmax=0;
roisize=p.ROIxy;
roisizeh=round(1.5*(p.ROIxy-1)/2); %create extra space if we need to shift;
rsr=-roisizeh:roisizeh;
filelist=p.filelist;
if ~iscell(filelist) %single file
    filelist={filelist};
end
b=[];
ht=uitab(p.tabgroup,'Title','Files');
tg=uitabgroup(ht);
for k=1:length(filelist)
    ax=axes(uitab(tg,'Title',num2str(k)));
    p.fileax(k)=ax;
    if isfield(p,'smap') && p.smap
        imstack=readfile_ome(filelist{k});
    else
        imstack=readfile_tif(filelist{k});
    end
    
    if p.emgain
        imstack=imstack(:,end:-1:1,:);
    end
    
    if isfield(p,'framerangeuse')
        imstack=imstack(:,:,p.framerangeuse(1):p.framerangeuse(end));
    end
    
     
    imstack=imstack-min(imstack(:)); %fast fix for offset;
  
    mim=max(imstack,[],3);
    mim=filter2(h,mim);
    maxima=maximumfindcall(mim);
%     figure(88);
    imagesc(ax,mim);
    axis(ax,'equal');
    axis(ax,'off')
    title(ax,'Maximum intensity projection')
    int=maxima(:,3);
    try
    mimc=mim(roisize:end-roisize,roisize:end-roisize);
    mmed=quantile(mimc(:),0.3);
    imt=mimc(mimc<mmed);
        sm=sort(int);
    mv=mean(sm(end-5:end));
%     cutoff=mean(imt(:))+max(3*std(imt(:)),(mv-mean(imt(:)))/5);
    cutoff=mean(imt(:))+max(2.5*std(imt(:)),(mv-mean(imt(:)))/15);
%     iq=quantile(int,0.5);
%    cutoff= mean(int(int<iq))+3*std(int(int<iq));
    
    
    catch
        cutoff=quantile(mimc(:),.95);
    end
%     cutoff=(quantile(mimc(:),0.8)+quantile(mimc(:),0.99))/2;

%    cutoff= (mv+quantile(mimc(:),0.5))/2;
    if any(int>cutoff)
        maxima=maxima(int>cutoff,:);
    else
        [~,indm]=max(int);
        maxima=maxima(indm,:);
    end
    
    if isfield(p,'beadpos') %passed on externally
        maxima=round(p.beadpos{k});
    end
    
    
    hold (ax,'on')
    plot(ax,maxima(:,1),maxima(:,2),'ko')
    hold (ax,'off')
    drawnow
    numframes=size(imstack,3);
    bind=length(b)+size(maxima,1);
%     bold=size(maxima,1);
    for l=1:size(maxima,1)
        b(bind).loc.frames=(1:numframes)';
        b(bind).loc.filenumber=zeros(numframes,1)+k;
        b(bind).filenumber=k;
        b(bind).pos=maxima(l,1:2);
        try
            b(bind).stack.image=imstack(b(bind).pos(2)+rsr,b(bind).pos(1)+rsr,:);
            b(bind).stack.framerange=1:numframes;
            b(bind).isstack=true;
            
        catch err
            b(bind).isstack=false;
%             err
        end
        if isfield(p,'files')
            b(bind).roi=p.files(k).info.roi;
        else
            b(bind).roi=[0 0 size(imstack,1) size(imstack,2)];
        end
        bind=bind-1;
    end
    fmax=max(fmax,numframes);
end
b=b([b(:).isstack]);

p.fminmax=[1 fmax];

        if isfield(p,'files')
            p.cam_pixelsize_um=p.files(k).info.cam_pixelsize_um;
        else
            p.cam_pixelsize_um=[1 1]/1000; %?????
        end      

p.pathhere=fileparts(filelist{1});
end


