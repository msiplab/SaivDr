function cases = atomcases()
%ATOMCASES Test cases of the atomic images (atomicimshow)
cases = {};
combos = {
    % nComps nChs   dec    ppo    nLevels patchsize
    {1,     [3 3], [2 2], [2 2], 1,      [16 16]}
    {1,     [4 4], [2 2], [2 2], 2,      [16 16]}
    {3,     [3 3], [2 2], [0 2], 2,      [16 16]}
    };
str = @(v) strrep(num2str(v),' ','');
for iCombo = 1:numel(combos)
    v = combos{iCombo};
    p = struct('NumberOfComponents',v{1},'NumberOfChannels',v{2},...
        'DecimationFactor',v{3},'PolyPhaseOrder',v{4},'NumberOfLevels',v{5},...
        'NumberOfVanishingMoments',[1 1],'Mode','Synthesizer');
    name = sprintf('Atoms2d_cmp%d_nch%s_dec%s_ppo%s_lv%d',v{1},str(v{2}),str(v{3}),...
        str(v{4}),v{5});
    cases{end+1,1} = struct('name',name,'params',p,'patchsize',v{6}); %#ok<AGROW>
end
end
