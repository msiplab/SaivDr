function c = mkcase(name,layer,params,inputs,outkinds,varargin)
%MKCASE Create a test case of a layer
%
%   inputs   : n x 2 cell array of {kind, size}
%   outkinds : cell array of the kinds of the outputs
%
%   Options: 'DataType' ('double'), 'Learnable' (false), 'RandomMus' (false),
%            'Backward' (true)
p = inputParser;
addParameter(p,'DataType','double')
addParameter(p,'Learnable',false)
addParameter(p,'RandomMus',false)
addParameter(p,'Backward',true)
parse(p,varargin{:})
c = struct('name',[name '_' p.Results.DataType],'layer',layer,'params',params,...
    'inputs',{inputs},'outkinds',{outkinds},'dtype',p.Results.DataType,...
    'learnable',p.Results.Learnable,'randmus',p.Results.RandomMus,...
    'backward',p.Results.Backward);
end
