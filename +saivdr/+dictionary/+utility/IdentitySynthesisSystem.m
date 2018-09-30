classdef IdentitySynthesisSystem < saivdr.dictionary.AbstSynthesisSystem
    %IDENTITYSYNTHESISSYSTEM Identity synthesis system
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2018, Shogo MURAMATSU
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
        
        function obj = IdentitySynthesisSystem(varargin)
            obj.FrameBound = 1;
        end
        
    end
    
    methods (Access = protected)
        
        function recImg = stepImpl(~,coefs,scales)
            recImg = reshape(coefs,scales);
        end
        
    end
  
end