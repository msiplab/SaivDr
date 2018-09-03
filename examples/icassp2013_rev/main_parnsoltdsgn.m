% MAIN_PARNSOLTDSGN NSOLT design process 
%
% This script executes design process of NSOLT with frequency 
% domain specification.
%
% Requirements: MATLAB R2015b, Global optimization toolbox
%
% Copyright (c) 2014-2016, Shogo MURAMATSU
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

%% Parameter settings
params.Display = 'final';
params.useParallel = 'always';
params.plotFcn = @gaplotbestf;
params.populationSize = 16;
params.eliteCount = 2;
params.generations = 40;
params.stallGenLimit = 20;
params.mutationFcn = @mutationgaussian;
params.nVm = 1;
params.dec = [ 2 2 ];
params.chs = [ 5 2 ];

%% Frequency domain specification
P =  ones(8); % Pass band value
S = -ones(8); % Stop band value
T = zeros(8); % Transition band value
A = zeros(8);
for iCol = 1:8
    for iRow = 1:8
        if iCol>(9-iRow)
            A(iRow,iCol) = 0;
        else
            A(iRow,iCol) = 1;
        end
    end
end
specPassStopBand(:,:,1) =  [
    S S S S S S S S S S S S S S S S ;
    S S S S S S S S S S S S S S S S ;
    S S S S S S S S S S S S S S S S ;
    S S S T T T T T T T T T T S S S ;
    S S S T T T T T T T T T T S S S ;
    S S S T T P P P P P P T T S S S ;
    S S S T T P P P P P P T T S S S ;
    S S S T T P P P P P P T T S S S ;
    S S S T T P P P P P P T T S S S ;
    S S S T T P P P P P P T T S S S ;
    S S S T T P P P P P P T T S S S ;
    S S S T T T T T T T T T T S S S ;
    S S S T T T T T T T T T T S S S ;
    S S S S S S S S S S S S S S S S ;
    S S S S S S S S S S S S S S S S ;
    S S S S S S S S S S S S S S S S ]; % E
specPassStopBand(:,:,2) = [
    T T T T T T T T T T T T T T T T ;
    T P P P P P P T T S S S S S S T ;
    T P P P P P P T T S S S S S S T ;
    T T T T T T T T T S S S S S S T ;
    T T T T T T T T T S S S S S S T ;
    S S S S S S S S S S S S S S S S ;
    S S S S S S S S S S S S S S S S ;
    S S S S S S S S S S S S S S S S ;
    S S S S S S S S S S S S S S S S ;
    S S S S S S S S S S S S S S S S ;
    S S S S S S S S S S S S S S S S ;
    T S S S S S S T T T T T T T T T ;
    T S S S S S S T T T T T T T T T ;
    T S S S S S S T T P P P P P P T ;
    T S S S S S S T T P P P P P P T ;
    T T T T T T T T T T T T T T T T ];
%
specPassStopBand(:,:,3) = flipud(specPassStopBand(:,:,2)); % E
specPassStopBand(:,:,4) = specPassStopBand(:,:,2).'; % E
specPassStopBand(:,:,5) = fliplr(specPassStopBand(:,:,4)); % E
specPassStopBand(:,:,6) = circshift(specPassStopBand(:,:,1),[0 64]); % O
specPassStopBand(:,:,7) = specPassStopBand(:,:,6).'; % O
%
params.spec = specPassStopBand;

%% Optimization
nReps = 4;
startOrd = 2; % Min. polyphase order
endOrd   = 4; % Max. polyphase order
stepOrd = 2;
for iRep = 1:nReps
    for swmus = 0:1 
        params.swmus = swmus;
        for ord = startOrd:stepOrd:endOrd
            params.ord = [ ord ord ];
            fprintf('dec=[%d %d], ch=[%d %d], ord=[%d %d]\n',...
                params.dec(1),params.dec(2),...
                params.chs(1),params.chs(2),...
                params.ord(1),params.ord(2));
            support.fcn_updatensolt(params); % Design and update function
        end
    end
end
