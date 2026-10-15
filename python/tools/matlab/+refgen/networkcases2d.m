function cases = networkcases2d()
%NETWORKCASES2D Test cases of fcn_creatensoltlgraph2d
import refgen.mknetcase
cases = {};
str = @(v) strrep(num2str(v),' ','');
% Representative combinations; each mode is tested with all of them
combos = {
    % nComps nChs   dec    ppo    nLevels  nVm
    {1,     [2 2], [2 2], [0 0], 1,       [1 1]}
    {1,     [3 3], [2 2], [2 2], 1,       [0 0]}
    {1,     [4 4], [2 2], [2 0], 2,       [1 0]}
    {1,     [4 4], [2 2], [0 2], 3,       [1 1]}
    {1,     [5 5], [2 4], [2 2], 1,       [1 1]}
    {3,     [3 3], [2 2], [2 2], 2,       [1 1]}
    {3,     [4 4], [1 2], [0 2], 1,       [0 1]}
    };
for mode = {'Whole','Analyzer','Synthesizer'}
    for iCombo = 1:numel(combos)
        v = combos{iCombo};
        p = struct('NumberOfComponents',v{1},'NumberOfChannels',v{2},...
            'DecimationFactor',v{3},'PolyPhaseOrder',v{4},'NumberOfLevels',v{5},...
            'NumberOfVanishingMoments',v{6},'Mode',mode{1});
        name = sprintf('Nsolt2d_%s_cmp%d_nch%s_dec%s_ppo%s_lv%d_vm%s',lower(mode{1}),...
            v{1},str(v{2}),str(v{3}),str(v{4}),v{5},str(v{6}));
        inputSize = 2*v{3}.^v{5}.*[2 1]; % divisible at every level
        cases{end+1,1} = mknetcase(name,2,p,inputSize); %#ok<AGROW>
    end
end
p = struct('NumberOfComponents',1,'NumberOfChannels',[4 4],'DecimationFactor',[2 2],...
    'PolyPhaseOrder',[2 2],'NumberOfLevels',2,'NumberOfVanishingMoments',[1 1],'Mode','Whole');
cases{end+1,1} = mknetcase('Nsolt2d_whole_cmp1_nch44_dec22_ppo22_lv2_vm11',2,p,[16 16],...
    'DataType','single');
end
