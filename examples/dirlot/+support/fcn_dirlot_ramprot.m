function [dirlotNorm0woScl,dctNorm0woScl,phixd,arrayCard] = ...
    fcn_dirlot_ramprot(params)
% FCN_DIRLOT_RAMPROT
%
% SVN identifier:
% $Id: fcn_dirlot_ramprot.m 683 2015-05-29 08:22:13Z sho $
%
% Requirements: MATLAB R2015b
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
import saivdr.dictionary.utility.Direction
eps = params.eps;
dim = params.dim;
location = params.location;

%%
dec   = params.dec;% Decimation factor
ord   = params.ord;
phi   = params.phi;% Deformation parameter
border = ceil(ord/2);

arrayCard = (dim(1)-2*dec*border)*(dim(2)-2*dec*border)*(1-1/dec^2);

if isempty(phi)
    filterName = sprintf(...
        '%s/nsgenlot_d%dx%d_o%d+%d_v2.mat', ...
        location, dec, dec, ord, ord);
    offset = 0;
else
    filterName = sprintf(...
        '%s/dirlot_d%dx%d_o%d+%d_tvm%06.2f.mat', ...
        location, dec, dec, ord, ord, phi);
    offset = round(mod(phi,1)*100)/100;
end
display(filterName)
tmp = load(filterName,'lppufb');
lppufb = saivdr.dictionary.utility.fcn_upgrade(tmp.lppufb);
release(lppufb)
set(lppufb,'OutputMode','ParameterMatrixSet')
import saivdr.dictionary.nsgenlotx.NsGenLotFactory
dct = NsGenLotFactory.createLpPuFb2dSystem(...
    'NumberOfVanishingMoments',0,...
    'DecimationFactor',[dec dec],...
    'OutputMode','ParameterMatrixSet');

%%
import saivdr.dictionary.nsgenlotx.NsGenLotUtility
import saivdr.dictionary.nsoltx.NsoltFactory
phixd = -45:135;
phixdoff = phixd+offset;
phixd = reshape([phixd(:) phixdoff(:)].',2*length(phixd),1);
dirlotNorm0woScl = zeros(1,length(phixd));
dctNorm0woScl = zeros(1,length(phixd));
parfor iPhixd = 1:length(phixd)
    
    srcImg = NsGenLotUtility.trendSurface(phixd(iPhixd),dim);
    
    % DirLot
    fdirlot = NsoltFactory.createAnalysis2dSystem(lppufb,...
        'BoundaryOperation','circular');
    [coefsLot,scalesLot] = step(fdirlot,srcImg,1);
            
    eIdx = prod(scalesLot(1,:));
    for iSubband = 2:length(scalesLot)
        sIdx = eIdx + 1;
        eIdx = sIdx + prod(scalesLot(iSubband,:))-1;
        coef = reshape(coefsLot(sIdx:eIdx),scalesLot(iSubband,:));
        % ignore the boundary coefs.
        coef = coef(border+1:end-border,border+1:end-border);
        subCoef = abs(coef);
        dirlotNorm0woScl(iPhixd) = dirlotNorm0woScl(iPhixd)...
            + length(find(subCoef(:)>eps));
    end

    % Dct
    fdct = NsoltFactory.createAnalysis2dSystem(dct,...
        'BoundaryOperation','circular');    
    [coefsDct,scalesDct] = step(fdct,srcImg,1);

    eIdx = prod(scalesLot(1,:));
    for iSubband = 2:length(scalesDct)
        sIdx = eIdx + 1;
        eIdx = sIdx + prod(scalesLot(iSubband,:))-1; 
        coef = reshape(coefsDct(sIdx:eIdx),scalesLot(iSubband,:));
        % ignore the boundary coefs.
        coef = coef(border+1:end-border,border+1:end-border); 
        subCoef = abs(coef);
        dctNorm0woScl(iPhixd) = dctNorm0woScl(iPhixd)...
            + length(find(subCoef(:)>eps));
    end
    
end
