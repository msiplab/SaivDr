function cases = nsoltxcases()
%NSOLTXCASES Test cases of saivdr_export_nsoltx
%
%   A pre-designed filter bank (examples/icassp2014) and random ones are
%   compared through saivdr.dictionary.generalfb.Analysis2dSystem and
%   Synthesis2dSystem with frequency-domain filtering, as in
%   examples/rgbdemos.
root = fileparts(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath'))))));
designed = fullfile(root,'examples','icassp2014','results',...
    'nsolt_d2x2_c4+4_o4+4_v1_l4_n2048_lena128.mat');
cases = {
    struct('name','Nsoltx_designed_lena128_lv1','source',designed,'nLevels',1,'size',[32 32])
    struct('name','Nsoltx_designed_lena128_lv4','source',designed,'nLevels',4,'size',[128 128])
    };
combos = {
    % nChs   ppo    nVm nLevels size
    {[2 2], [0 0], 0,  1,      [16 16]}
    {[3 3], [2 2], 1,  2,      [32 32]}
    {[4 4], [2 0], 1,  1,      [16 16]}
    {[4 4], [0 4], 0,  3,      [64 64]}
    };
str = @(v) strrep(num2str(v),' ','');
for i = 1:numel(combos)
    v = combos{i};
    cases{end+1,1} = struct('name',sprintf('Nsoltx_random_nch%s_ppo%s_vm%d_lv%d',...
        str(v{1}),str(v{2}),v{3},v{4}),'source','random','nChs',v{1},'ppo',v{2},...
        'nVm',v{3},'nLevels',v{4},'size',v{5}); %#ok<AGROW>
end
end
