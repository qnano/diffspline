clear all
fn="C:/code/jelmer/data/zstack-1594737820-1.tif";

tiff_info = imfinfo(fn); % return tiff structure, one element per image
tiff_all = imread(fn, 1) ; % read in first image
%concatenate each successive tiff to tiff_stack
for ii = 2 : size(tiff_info, 1)
    temp_tiff = imread(fn, ii);
    tiff_all = cat(3 , tiff_all, temp_tiff);
end


mat=load("C:/code/jelmer/data/init_abb.mat");
abb=double(mat.aberrations_amp);

fn="C:/code/ExamplePhaseRetrieval/data/figure2/18rms_astig50_2304.mat";
mat=load(fn);

act=linspace(-0.2,0.2,7);
z=linspace(-2000,2000,41);
all=5;
aberrarray=zeros(27,all);

for i=1:all
    disp(i);
    tiff_s=tiff_all(:,:,(i-1)*9+1:(i)*9);
    
    %zind=10:2:31;
    %zind=zind;
    tiff_stack=tiff_s(:,:,:);

    tiff_stack=(tiff_stack-103.1)*0.49;

    z=linspace(-1000,1000,41);
    [Mx,My,Mz]=size(tiff_stack);
    parameters=mat.parameters;
    parameters.fitmodel='aberrations';

        
    [Mx,My,Mz]=size(tiff_stack);
    
    parameters=mat.parameters;
    parameters.Mx=Mx;
    parameters.My=My;
    parameters.Mz=Mz;
    parameters.pixelsize=81.25;
    parameters.bead=true;
    parameters.zrange=[-1000,1000];
    parameters.beaddiameter=500;
    parameters.zspread=[-300,300];
    %start the vector fitter class
    a=vector_fitter_class;
    %set parameters
    a.parameters=parameters;
    a.parameters.fitmodel= 'aberrations';
    %set max radial degree
    a.maxn=6;
    a.parameters.xrange = a.parameters.pixelsize*a.parameters.Mx/2;
    a.parameters.yrange = a.parameters.pixelsize*a.parameters.My/2;
    %set z range
    %you also can set other parameters. you can refer the class code 

    %calibrate PSF model
    [aberrations_amp]=a.calibrate_PSF_ampaberration_phase(tiff_stack);
    aberrarray(:,i)=aberrations_amp(:,3);
end

if false
    x=linspace(-50,50,5);
    subplot(2,2,1)
    plot(x,aberrarray(:,1:7)'-aberrarray(:,4)');
    subplot(2,2,2)
    plot(x,aberrarray(:,8:14)'-aberrarray(:,11)');
    subplot(2,2,3)
    plot(x,aberrarray(:,15:21)'-aberrarray(:,18)');
    subplot(2,2,4)
    plot(x,aberrarray(:,22:28)'-aberrarray(:,25)');


    x=linspace(-100,100,7);
    subplot(2,2,1)
    plot(x,aberrarray(:,1:7)');
    subplot(2,2,2)
    plot(x,aberrarray(:,8:14)');
    subplot(2,2,3)
    plot(x,aberrarray(:,15:21)');
    subplot(2,2,4)
    plot(x,aberrarray(:,22:28)');
end