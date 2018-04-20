function [Bsep,A,D] = fcn_ksvdsdiclrn2(params)

if nargin < 1
    %imgName = 'barbara128';
    srcImg  = load_testimg(imgName);
    blocksize = 8; % size of block to operate on
    odctsize  = 13;
    trainnum = 10000;
    isMonitoring = true;
    verbose = 'irt';
else
    %imgName = params.imgName;
    srcImg = params.srcImg;
    blocksize = params.blocksize;
    odctsize = params.odctsize;
    trainnum = params.trainnum;
    isMonitoring = params.isMonitoring;
    verbose = params.verbose;
end

%%
msgdelta = 5;
x = im2double(srcImg);
%odctsize = ceil(blocksize*sqrt(redundancy));
params.dictsize = odctsize^2;

%%
params.basedict{1} = odctdict(blocksize,odctsize);   % the separable base dictionary -
params.basedict{2} = odctdict(blocksize,odctsize);   % basedict{i} is the base dictionary
params.initA = speye(params.dictsize); % initial A matrix (identity)
%params.maxval = 255;                  % maximum sample intensity value
%trainnum = 10000;                      % number of training blocks
%params.iternum = 15;                  % number of training iterations
params.Tdict = 8;                      % sparsity of each trained atom
params.Tdata = 8;                      % number of coefficients

%%%% create training data %%%

p = ndims(x);
ids = cell(p,1);
[ids{:}] = reggrid(size(x)-blocksize+1, trainnum, 'eqdist');
params.data = sampgrid(x,blocksize,ids{:});

% remove dc in blocks to conserve memory %
shiftsize = 2000;
for i = 1:shiftsize:size(params.data,2)
  blockids = i : min(i+shiftsize-1,size(params.data,2));
  params.data(:,blockids) = remove_dc(params.data(:,blockids),'columns');
end

%%%%% KSVDS training %%%%%

if (msgdelta>0)
  disp('Sparse K-SVD training...');
end
Bsep = params.basedict;
A = ksvds(params,verbose,msgdelta);
D = dictsep(Bsep,A,speye(size(A,2)));

%%
if isMonitoring
    for iRow = 1:odctsize
        for iCol = 1:odctsize
            idx = (iRow-1)*odctsize+iCol;
            subplot(odctsize,odctsize,idx);
            subimage(reshape(D(:,idx),blocksize,blocksize)+0.5)
            axis off
        end
    end
    drawnow
end

