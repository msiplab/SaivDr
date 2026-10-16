function c = mknetcase(name,ndims_,params,inputSize,varargin)
%MKNETCASE Create a test case of a network
p = inputParser;
addParameter(p,'DataType','double')
addParameter(p,'BatchSize',2)
parse(p,varargin{:})
c = struct('name',[name '_' p.Results.DataType],'ndims',ndims_,'params',params,...
    'inputSize',inputSize,'dtype',p.Results.DataType,'batchSize',p.Results.BatchSize);
end
