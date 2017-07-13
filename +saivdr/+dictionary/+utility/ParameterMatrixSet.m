classdef ParameterMatrixSet < saivdr.dictionary.utility.ParameterMatrixContainer
    %PARAMETERMATRIXSET Parameter matrix set (deprecated)
    %
    % This class was renamed ParameterMatrixContainer.
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2016, Shogo MURAMATSU
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
    
    methods
        
        function obj = ParameterMatrixSet(varargin)
            obj = obj@saivdr.dictionary.utility.ParameterMatrixContainer(varargin{:});
            warning('This class is deprecated.')
        end
        
    end

end
