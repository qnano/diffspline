clear all

Iall=zeros(15,1);
for i=1:15
    Iall(i)=1e5*(0.7)^i;
end
bg=3;

k=6;

mat=load('C:/code/ExamplePhaseRetrieval/data/rmsdata/36rms-36rms_lerrmoremore_nobead.mat');
err3636=(mat.err);
%bias=mean(err3636(:,k,:),3);
s=zeros(15,8);
for i=1:8
    s(:,i)=sqrt(mean(err3636(:,k,(i-1)*16+1:i*16).*err3636(:,k,(i-1)*16+1:i*16),3));
    %s(:,i)=(std(err3636(:,k,(i-1)*16+1:i*16),0,3));
end
rmse=sqrt(mean(err3636(:,k,:).*err3636(:,k,:),3));
%stderr=(std(err3636(:,k,:),0,3));
errorbar(Iall,rmse,std(s,0,2));

hold on

mat=load('C:/code/ExamplePhaseRetrieval/data/rmsdata/36rms-18rms_lerrmoremore_nobead.mat');
err3618=(mat.err);
%stderr=(std(err3618(:,k,:),0,3));
rmse=sqrt(mean(err3618(:,k,:).*err3618(:,k,:),3));
%bias=mean(err3618(:,k,:),3);
s=zeros(15,8);
for i=1:8
    s(:,i)=sqrt(mean(err3618(:,k,(i-1)*16+1:i*16).*err3618(:,k,(i-1)*16+1:i*16),3));
    %s(:,i)=(mean(err3618(:,k,(i-1)*16+1:i*16),3));
end
errorbar(Iall,rmse,std(s,0,2));
%errorbar(Iall,ones(10,1),std(s,0,2));
hold on

mat=load('C:/code/ExamplePhaseRetrieval/data/rmsdata/18rms-18rms_lerrmoremore_nobead.mat');
err1818=(mat.err);
rmse=sqrt(mean(err1818(:,k,:).*err1818(:,k,:),3));
%stderr=(std(err1818(:,k,:),0,3));
bias=mean(err1818(:,k,:),3);
s=zeros(15,8);
for i=1:8
    s(:,i)=sqrt(mean(err1818(:,k,(i-1)*16+1:i*16).*err1818(:,k,(i-1)*16+1:i*16),3));
    %s(:,i)=(std(err1818(:,k,(i-1)*16+1:i*16),0,3));
end
errorbar(Iall,rmse,std(s,0,2));

hold on

mat=load('C:/code/ExamplePhaseRetrieval/data/rmsdata/36rmscrlb_nobead.mat');
crlb36=mat.CRLBarr;
plot(Iall,crlb36(:,k),"--+");

hold on

mat=load('C:/code/ExamplePhaseRetrieval/data/rmsdata/18rmscrlb_nobead.mat');
crlb18=mat.CRLBarr;
plot(Iall,crlb18(:,k),"--+");



set(gca, 'XScale', 'log', 'YScale', 'log')
l=legend({'36rms data+36rms model','36rms data+18rms model','18rms data+18rms model','36rms CRLB','18rms CRLB'},'Location','northeast');
l.FontSize = 6;
xlabel("intensity (photons)");
ylabel("\sigma_{x}");
