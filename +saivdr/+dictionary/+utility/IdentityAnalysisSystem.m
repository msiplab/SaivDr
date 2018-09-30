classdef IdentityAnalysisSystem < saivdr.dictionary.AbstAnalysisSystem
    %IDENTITYANALYSISSYSTEM Identity analysis system
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
   
    methods (Access = protected)
        
        function [coefs,scales] = stepImpl(~,srcImg)
            coefs = srcImg(:);
            scales = size(srcImg);            
        end
    end
    
end
