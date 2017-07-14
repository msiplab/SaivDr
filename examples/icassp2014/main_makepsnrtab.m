%MAIN_MAKEPSNRTAB Generate a table of PSNRs
%
% Requirements: MATLAB R2015b
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

%% General parameters
isEdit = ~true;
params.isMonitoring = false;

%% Parameter settings for KSVDS
support.fcn_setup_ksvdsbox
params.Tdict = 6;
params.Tdata = 8;
params.TdataImpl = 8;
params.blocksize = 8;
params.odctsize  = 13;
params.useParallel = true;

%% Parameter settings for NSOLT
params.dec = [ 2 2 ];
params.ord = [ 4 4 ];
params.nCoefs = 2048;
params.nCoefsImpl = 2048;
params.index = 'end';

%% Condition setting 
imgSet = { 'goldhill128', 'lena128', 'barbara128', 'baboon128' }; 
imgNames = { 'goldhill', 'lena', 'barbara', 'baboon' }; 
vmSet = { 1 };
chSet = { [ 4 4 ], [ 5 3 ], [ 6 2 ] };
lvSet = { 1,2,3,4,5 };

%%
table = cell((length(lvSet)*length(vmSet)*length(chSet))+4,length(imgSet)+1);
psnrs = zeros(1,length(vmSet)*length(chSet)+1);
for iCol = 1:length(imgSet)+1
    iRow = 1;
    if iCol == 1
        table{iRow,iCol} = '\multicolumn{2}{|c||}{Dictionary}';
    else
        iImg = iCol - 1;
        params.imgName = imgSet{iImg};
        table{iRow,iCol} = sprintf('{\\it %s}',imgNames{iImg});
    end
    iRow = iRow + 1;
    % ompksvds
    if iCol == 1
        table{iRow,iCol} = '\multicolumn{2}{|c||}{Sparse K-SVD ($\mathcal{R}=\frac{169}{64}$) }';
    else
        if isEdit
            psnr = 0;
        else
            psnr = support.fcn_ompsksvds(params);
        end
        psnrs(iRow-1) = psnr;
        table{iRow,iCol} = sprintf('%6.2f',psnr);
    end
    iRow = iRow + 1;    
    % ihtnsolt
    if iCol == 1
        table{iRow,1} = '\multicolumn{2}{|c||}{MS-NSOLT ($\mathcal{R}<\frac{8-1}{4-1}$)}';
    else
        table{iRow,iCol} = '';
    end
    iRow = iRow + 1;    
    if iCol == 1
        table{iRow,1} = '$p_{\rm s}+p_{\rm a}$ & $\tau$';
    else
        table{iRow,iCol} = '';
    end
    iRow = iRow + 1;
    for iVm = 1:length(vmSet)
         params.nVm = vmSet{iVm};
          if iCol == 1
              table{iRow,iCol} = '';
         %               table{iRow,iCol} = sprintf('%d & ',params.nVm);
          end

        for iCh = 1:length(chSet)
            params.chs = chSet{iCh};
            if iCol == 1
                if iCh == 1
                    table{iRow,iCol} = sprintf('%s $%d+%d$',table{iRow,iCol},...
                        params.chs(1),params.chs(2));
                else
                    table{iRow,iCol} = sprintf('$%d+%d$',...
                        params.chs(1),params.chs(2));                    
                end
            end
            %
            psnrsTmp  = cell(1,length(lvSet));
            paramsTmp = cell(1,length(lvSet));
            parfor iLv = 1:length(lvSet)
                paramsTmp{iLv} = params;
                paramsTmp{iLv}.nLevels = lvSet{iLv};
                if iCol ~= 1
                    if isEdit
                        psnrsTmp{iLv} = 0;
                    else
                        psnrsTmp{iLv} =  support.fcn_ihtnsolt(paramsTmp{iLv});
                    end
                end
            end            
            %
            for iLv = 1:length(lvSet)
                if iCol == 1
                    if iLv == 1 
                        table{iRow,iCol} = sprintf('%s & %d',table{iRow,iCol},...
                            paramsTmp{iLv}.nLevels);
                    else
                        table{iRow,iCol} = sprintf('   & %d',...
                            paramsTmp{iLv}.nLevels);                        
                    end
                else
                    psnrs(iRow-1) = psnrsTmp{iLv};
                    table{iRow,iCol} = sprintf('%6.2f',psnrsTmp{iLv});
                end
                iRow = iRow + 1;
            end
        end
    end
    if iCol ~=1
        [~,iRowMax] = max(psnrs(:));
        table{iRowMax+1,iCol} = [ '{\bf ' table{iRowMax+1,iCol} '}' ];
    end
end

%%
sw = StringWriter();
%
sw.add('\begin{tabular}{|c|c||')
for iCol = 1:size(table,2)
    sw.add('c|');
end
sw.addcr('} \hline')
%
for iRow = 1:size(table,1)
    for iCol = 1:size(table,2)-1
        sw.add(table{iRow,iCol});
        sw.add(' & ');
    end
    sw.add(table{iRow,end});
    if iRow == 1
        sw.addcr(' \\ \hline\hline')
    else
        sw.addcr(' \\ \hline')
    end
end
%
sw.add('\end{tabular}')

%%
disp(sw)
