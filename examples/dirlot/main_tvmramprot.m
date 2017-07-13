% MAIN_TVMRAMPROT Script for evaluationg TVM characteristics of DirLOT
%
% Shogo Muramatsu, Dandan Han, Tomoya Kobayashi and Hisakazu Kikuchi: 
%  ''Directional Lapped Orthogonal Transform: Theory and Design,'' 
%  IEEE Trans. on Image Processing, Vol.21, No.5, pp.2434-2448, May 2012.
%  (DOI: 10.1109/TIP.2011.2182055)
%
% Requirements: MATLAB R2015b, Global optimization toolbox
%
% Copyright (c) 2014-2017, Shogo MURAMATSU
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
isDisplay = false;
if isDisplay 
    hfig = figure;
end

%% Ramp Rotation
eps = 1e-15;
dim = [ 32 32 ];

arrayCard = ((dim(1)-4)*(dim(2)-4))*3/4;

%% DirLOT

phid  = -45:134;
phixd = -45:134;
arrayNorm0woScl = zeros(length(phid),length(phixd));
for iPhid = 1:length(phid)
    
    phi = phid(iPhid)*pi/180;
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
    
    dirlot = support.fcn_ezdirlottvm(phi,ss,theta);

    if isDisplay
        figure(hfig)
        atmimshow(dirlot)
        drawnow
    end
    
    subbandNorm0woScl = zeros(1,length(phixd));
    parfor iPhixd = 1:length(phixd)
        phix = phixd(iPhixd)*pi/180;
        srcImg = saivdr.dictionary.nsgenlotx.NsGenLotUtility.trendSurface(...
            phixd(iPhixd),dim);
        
        % DirLot
        fdirlot = saivdr.dictionary.nsoltx.NsoltFactory.createAnalysis2dSystem(...
            dirlot,...
            'BoundaryOperation','circular');
        nLevels = 1;
        [coefsLot,scalesLot] = step(fdirlot,srcImg,nLevels);
        
        sIdx = 1;
        eIdx = prod(scalesLot(1,:));
        for iSubband = 2:size(scalesLot,1)
            sIdx = eIdx+1;
            eIdx = sIdx+prod(scalesLot(iSubband,:))-1;
            coef = reshape(coefsLot(sIdx:eIdx),scalesLot(iSubband,:));
            coef = coef(2:end-1,2:end-1); % ignore the boundary coefs.
            subCoef = abs(coef);
            subbandNorm0woScl(iPhixd) = subbandNorm0woScl(iPhixd)...
                + length(find(subCoef(:)>eps));
        end
    end
    arrayNorm0woScl(iPhid,:) = subbandNorm0woScl(:).';
    
    disp(['phi = ' num2str(phid(iPhid))])
end

%%
save results/tvmramprot_results arrayNorm0woScl arrayCard
if isDisplay
    close(hfig)
end
