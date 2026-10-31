function cases = networkcases3d()
%NETWORKCASES3D Test cases of refgen.creatensoltlgraph3d
import refgen.mknetcase
cases = {};
str = @(v) strrep(num2str(v),' ','');
combos = {
    % nChs   dec      ppo      nLevels nVm
    {[4 4], [2 2 2], [0 0 0], 1,      [1 1]}
    {[4 4], [2 2 2], [2 2 2], 1,      [1 0]}
    {[5 5], [2 2 2], [2 0 2], 2,      [1 1]}
    {[4 4], [1 2 2], [0 2 0], 2,      [0 0]}
    };
for mode = {'Whole','Analyzer','Synthesizer'}
    for iCombo = 1:numel(combos)
        v = combos{iCombo};
        p = struct('NumberOfComponents',1,'NumberOfChannels',v{1},...
            'DecimationFactor',v{2},'PolyPhaseOrder',v{3},'NumberOfLevels',v{4},...
            'NumberOfVanishingMoments',v{5},'Mode',mode{1});
        name = sprintf('Nsolt3d_%s_cmp1_nch%s_dec%s_ppo%s_lv%d_vm%s',lower(mode{1}),...
            str(v{1}),str(v{2}),str(v{3}),v{4},str(v{5}));
        inputSize = 2*v{2}.^v{4};
        cases{end+1,1} = mknetcase(name,3,p,inputSize); %#ok<AGROW>
    end
end
end
