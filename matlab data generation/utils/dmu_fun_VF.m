function r=dmu_fun_VF(theta,parameters)

dz=-200;
a=model_biplane(theta(1),theta(2),theta(3),theta(4),theta(5),parameters);
mu1=a.mu;
dmu1=a.dmudtheta;

a=model_biplane(theta(1),theta(2),theta(3)+dz,theta(4),theta(5),parameters);
mu2=a.mu;
dmu2=a.dmudtheta;

r.mu1=mu1;
r.mu2=mu2;

r.dmu1=dmu1;
r.dmu2=dmu2;

end