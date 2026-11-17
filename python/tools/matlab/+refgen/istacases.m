function cases = istacases()
%ISTACASES Test cases of IstaImRestoration2d as in examples/rgbdemos
root = fileparts(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath'))))));
designed = fullfile(root,'examples','icassp2014','results',...
    'nsolt_d2x2_c4+4_o4+4_v1_l4_n2048_lena128.mat');
cases = {
    struct('name','Ista_db','dictionary',designed,'nLevels',2,'degradation','db',...
        'lambda',0.00185,'nsigma',5,'maxIter',20,'size',[64 64 3])
    struct('name','Ista_sr','dictionary',designed,'nLevels',2,'degradation','sr',...
        'lambda',0.00018,'nsigma',0,'maxIter',20,'size',[64 64 3])
    struct('name','Ista_ip','dictionary',designed,'nLevels',2,'degradation','ip',...
        'lambda',0.01,'nsigma',0,'maxIter',20,'size',[64 64 1])
    };
end
