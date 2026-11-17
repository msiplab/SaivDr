function cases = sparserepcases()
%SPARSEREPCASES Test cases of saivdr.sparserep.IterativeHardThresholding
root = fileparts(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath'))))));
designed = fullfile(root,'examples','icassp2014','results',...
    'nsolt_d2x2_c4+4_o4+4_v1_l4_n2048_lena128.mat');
cases = {
    struct('name','Iht_lena128_lv1_n128_it30','dictionary',designed,'nLevels',1,...
        'size',[32 32],'nCoefs',128,'maxIter',30)
    struct('name','Iht_lena128_lv2_n512_it1000','dictionary',designed,'nLevels',2,...
        'size',[64 64],'nCoefs',512,'maxIter',1000)
    };
end
