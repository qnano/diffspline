function r=dmu_fun_astig(x,y,z,I,bg,coeff1)

Npixels=35;

xc=x;
yc=y;
zc=z;
xstart=0;
ystart=0;
zstart=100;
spline_xsize = size(coeff1,1);
spline_ysize = size(coeff1,2);
spline_zsize = size(coeff1,3);


[delta_f,delta_dxf,delta_ddxf,delta_dyf,delta_ddyf,delta_dzf,delta_ddzf]=computeDelta3Dj_v2(single(xc),single(yc),single(zc));

mu1=zeros([Npixels,Npixels]);
dxmu1=zeros([Npixels,Npixels]);
dymu1=zeros([Npixels,Npixels]);
dzmu1=zeros([Npixels,Npixels]);
dImu1=zeros([Npixels,Npixels]);
dbgmu1=zeros([Npixels,Npixels]);
ddxmu1=zeros([Npixels,Npixels]);
ddymu1=zeros([Npixels,Npixels]);
ddzmu1=zeros([Npixels,Npixels]);



R=1.0;

for ii = 0:Npixels-1
    for jj = 0:Npixels-1
         vx=ii+xstart;
         vy=jj+ystart;
         temp = fAt3Dj_v2(vx,vy,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,coeff1);
         model = temp*I*R+bg*R;
         mu1(ii+1,jj+1)=model;

        


         temp = fAt3Dj_v2(vx,vy,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,coeff1);
         model = temp;
         dImu1(ii+1,jj+1)=model*R;

         

         dbgmu1(ii+1,jj+1)=R;
        

         temp = fAt3Dj_v2(vx,vy,zstart,spline_xsize,spline_ysize,spline_zsize,delta_dxf,coeff1);
         model = temp*I*R;
         dxmu1(ii+1,jj+1)=model;
         
         temp = fAt3Dj_v2(vx,vy,zstart,spline_xsize,spline_ysize,spline_zsize,delta_ddxf,coeff1);
         model = temp*I*R;
         ddxmu1(ii+1,jj+1)=model;
         
         

         temp = fAt3Dj_v2(vx,vy,zstart,spline_xsize,spline_ysize,spline_zsize,delta_dyf,coeff1);
         model = temp*I*R;
         dymu1(ii+1,jj+1)=model;
         
         temp = fAt3Dj_v2(vx,vy,zstart,spline_xsize,spline_ysize,spline_zsize,delta_ddyf,coeff1);
         model = temp*I*R;
         ddymu1(ii+1,jj+1)=model;


         
        
         temp = fAt3Dj_v2(vx,vy,zstart,spline_xsize,spline_ysize,spline_zsize,delta_dzf,coeff1);
         model = temp*I*R;
         dzmu1(ii+1,jj+1)=model;
         
         temp = fAt3Dj_v2(vx,vy,zstart,spline_xsize,spline_ysize,spline_zsize,delta_ddzf,coeff1);
         model = temp*I*R;
         ddzmu1(ii+1,jj+1)=model;

         
    end
end
dmu1=zeros([Npixels,Npixels,5]);
ddmu1=zeros([Npixels,Npixels,5]);

r.mu1=mu1;

dmu1(:,:,1)=dxmu1;
dmu1(:,:,2)=dymu1;
dmu1(:,:,3)=dzmu1;
dmu1(:,:,4)=dImu1;
dmu1(:,:,5)=dbgmu1;

ddmu1(:,:,1)=ddxmu1;
ddmu1(:,:,2)=ddymu1;
ddmu1(:,:,3)=ddzmu1;
ddmu1(:,:,4)=0;
ddmu1(:,:,5)=0;
r.dmu1=dmu1;
r.ddmu1=ddmu1;


end