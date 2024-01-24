function [nrz,n,m] = getNrZernikeCoeffs(nz)

% maybe I don't really have to form n and m
n = []; m = [];
for ni = 0 : nz
    for mi = rem(ni,2) : 2 : ni
        if (ni == 0 && mi == 0), n = [n 0]; m = [m 0];
        else
            if (mi == 0), n = [n ni]; m = [m 0];
            else n = [n ni ni]; m = [m -mi mi];
            end
        end
    end
end

nrz = length(n);

[n,m] = zernfun2A(1:nrz,'Noll');
