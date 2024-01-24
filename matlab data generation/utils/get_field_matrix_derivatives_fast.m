%function [FieldMatrix,FieldMatrixDerivatives] =  get_field_matrix_derivatives_fast(parameters)
 function [FieldMatrix,FieldMatrixDerivatives] =...
     get_field_matrix_derivatives_fast(PupilMatrix,wavevector,wavevectorzimm,xemit,yemit,zemit,Ax,Bx,Dx)
% This function calculates the field matrix A_{jk}, which gives the j-th
% electric field component proportional to the k-th dipole vector
% component, as well as the derivatives of A_{jk} w.r.t. the xyz coordinates
% of the emitter and w.r.t. the emission wavelength lambda.
%
% copyright Sjoerd Stallinga, TU Delft, 2017
%
% parameters: NA, refractive indices of medium, wavelength (in nm), 
% nominal emitter position (in nm) with z-position from
% cover slip-medium interface, spot footprint (in nm), axial range (in nm),
% sampling in pupil with (even), sampling in image plane (odd), sampling in
% axial direction

% 2018 07 09, bug fix, MSie.
%PupilMatrix=parameters.PupilMatrix;
%wavevector=parameters.wavevector;
%wavevectorzimm=parameters.wavevectorzimm;


%xemit = parameters.xemit;
%yemit = parameters.yemit;
%zemit = parameters.zemit;

%Ax=parameters.Ax;
%Bx=parameters.Bx;
%Dx=parameters.Dx;

numders = 3;
% set number of relevant parameter derivatives

%   ZImage = zmin+DzImage/2:DzImage:zmax;
FieldMatrix = cell(2,3,1);
FieldMatrixDerivatives = cell(2,3,1,numders);

Wpos = xemit*wavevector{1}+yemit*wavevector{2}+zemit*wavevector{3};


PositionPhaseMask = exp(-1i*Wpos);
%   PositionPhaseMask = exp(1i*Wpos);
jz=1;
for itel = 1:2
for jtel = 1:3
    P=PositionPhaseMask.*PupilMatrix{itel,jtel};
% pupil functions and FT to matrix elements
%PupilFunction = PositionPhaseMask.*PupilMatrix{itel,jtel};


IntermediateImage = transpose(cztfunc(P,Ax,Bx,Dx));


FieldMatrix{itel,jtel,jz} = transpose(cztfunc(IntermediateImage,Ax,Bx,Dx));

% pupil functions for xy-derivatives and FT to matrix elements
for jder = 1:2
    PupilFunction = -1i*wavevector{jder}.*P;
    %         PupilFunction = 1i*wavevector{jder}.*PositionPhaseMask.*PupilMatrix{itel,jtel};
    IntermediateImage = transpose(cztfunc(PupilFunction,Ax,Bx,Dx));
    FieldMatrixDerivatives{itel,jtel,jz,jder} = transpose(cztfunc(IntermediateImage,Ax,Bx,Dx));
end

% pupil functions for z-derivative and FT to matrix elements

 zderindex = 3;

 PupilFunction = -1i*wavevectorzimm.*P;
%           PupilFunction = 1i*wavevectorzimm.*PositionPhaseMask.*PupilMatrix{itel,jtel};

IntermediateImage = transpose(cztfunc(PupilFunction,Ax,Bx,Dx));
FieldMatrixDerivatives{itel,jtel,jz,zderindex} = transpose(cztfunc(IntermediateImage,Ax,Bx,Dx));
  


end
end


end