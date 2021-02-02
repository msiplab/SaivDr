function [E, chkvmd] = fcn_ezdirlottvm(phi,ss,theta)
%FCN_EZDIRLOTTVM Function for simple construction of DirLOT 
%
% SVN identifier:
% $Id: fcn_ezdirlottvm.m 683 2015-05-29 08:22:13Z sho $
%
% Requirements: MATLAB R2015b, Global optimization toolbox
%
% Copyright (c) 2014-2015, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%

import saivdr.dictionary.utility.OrthonormalMatrixFactorizationSystem
import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
import saivdr.dictionary.utility.OrthogonalProjectionSystem
import saivdr.dictionary.nsgenlotx.NsGenLotFactory

phi = mod(phi+pi/4,pi)-pi/4;
dec = [2 2];
if phi >= -pi/4 && phi < pi/4 % d = x
    b = -[tan(phi) ; 1]/2;
    ord = [0 2];
elseif phi >= pi/4 && phi < 3*pi/4 % d = y
    b = -[1 ; cot(phi)]/2;
    ord = [2 0];
else
    error('Unexpected error occured.');
end

if nargin == 1 || (isempty(ss) && isempty(theta))
    if phi >=-pi/4 && phi < pi/4
        if phi <= 0
            ss = [1 0 1 0 1];
            theta = pi+phi;
        else
            ss = [0 0 1 0 1];
            theta = 0+phi;
        end
    elseif phi >=pi/4 && phi < 3*pi/4
        if phi <= pi/2
            ss = [1 0 0 0 0];
            theta = pi-phi;
        else
            ss = [0 0 0 0 0];
            theta = 0-phi;
        end
    else
        error('Unexped error occured.');
    end
end
omfs = OrthonormalMatrixFactorizationSystem();
omgs = OrthonormalMatrixGenerationSystem();
ops = OrthogonalProjectionSystem();

%
normb = norm(b);
lambda = (-1)^ss(1)*(acos(1-(normb^2)/2)+acos(normb/2));

% W0
W0 = diag([1 (-1)^ss(2)]);
[angW0,musW0] = step(omfs,W0);
mus(:,1) = musW0;
ang(1,1) = angW0;

% U0
[angP0,musP0] = step(ops,b);
P0 = step(omgs,angP0,musP0);
U0 = diag([1 (-1)^ss(3)])...
    *[cos(lambda) -sin(lambda) ; sin(lambda) cos(lambda) ]...
    *P0;
[angU0,musU0] = step(omfs,U0);
mus(:,2) = musU0;
ang(1,2) = angU0;

% U1 
a = [ 1 ; 0 ];
[angP1,musP1] = step(ops,-a-U0*b);
P1 = step(omgs,angP1,musP1);
U1 = diag([1 (-1)^ss(4)])*P1;
[angU1,musU1] = step(omfs,U1);
mus(:,3) = musU1;
ang(1,3) = angU1;

%U2
U2 = diag([1 (-1)^ss(5)])...
    *[cos(theta) -sin(theta) ; sin(theta) cos(theta) ];
[angU2,musU2] = step(omfs,U2);
mus(:,4) = musU2;
ang(1,4) = angU2;

chkvmd = norm(a+U1*a+U1*U0*b);
if chkvmd > 1e-15
    error('chkvmd violated (%g)',chkvmd)
end
E = NsGenLotFactory.createLpPuFb2dSystem(...
    'NumberOfVanishingMoments',0,...
    'DecimationFactor',dec,...
    'PolyPhaseOrder',ord,...
    'OutputMode','ParameterMatrixSet');
set(E,'Angles',ang);
set(E,'Mus', mus);
