classdef SubbandSpecification < matlab.System %~#codegen
    %SUBBANDSPECIFICATION Subband frequency specification
    %
    % This class evaluates the cost of given filterbanks in terms of the 
    % error energy between those filters and given ideal frequency 
    % specifications.
    %
    % SVN identifier:
    % $Id: SubbandSpecification.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2015, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/    
    %
    
    properties (Nontunable)
        DecimationFactor = [ 2 2 ]
        OutputMode = 'RegionOfSupport'; 
        % 
        SplitMode  = 'HalfSplit';
    end
    
    properties 
        Direction  = saivdr.dictionary.utility.Direction.VERTICAL
        Alpha      = 0.0
        Transition = 0.0
    end
    
    properties (Hidden, Transient)
        OutputModeSet = ...
            matlab.system.StringSet({...
            'RegionOfSupport',...
            'SubbandLocation',...
            'PassStopAssignment',...
            'AmplitudeSpecification',...
            'SubbandAssignment'});    
        SplitModeSet = ...
            matlab.system.StringSet({...
            'HalfSplit',...
            'QuarterSplit'});         
    end    
    
    properties (SetAccess = private, GetAccess = public)
        subbandIndices
        matrixD
        matrixE    = eye(2)
        matrixInvV = eye(2)
        matrixV    = eye(2)
    end
    
    methods
        
        % Constructor
        function obj = SubbandSpecification(varargin)
            setProperties(obj,nargin,varargin{:});
            obj.matrixD = diag(obj.DecimationFactor);
            obj.subbandIndices =  getSubbandIndices_(obj);
            update_(obj);
            if obj.Direction ~= saivdr.dictionary.utility.Direction.VERTICAL && ...
                    obj.Direction ~= saivdr.dictionary.utility.Direction.HORIZONTAL
                id = 'SaivDr:IllegalArgumentException';
                msg = sprintf(...
                    '%d: Direction must be either of 1 or 2.',...
                    obj.Direction);
                me = MException(id, msg);
                throw(me);
            end
        end
        
    end
    
    methods (Access = protected)
        
        function flag = isInactiveSubPropertyImpl(obj,propertyName)
            if strcmp(propertyName,'SplitMode')
                flag = ~(strcmp(obj.OutputMode,'SubbandLocation') || ...
                    strcmp(obj.OutputMode,'SubbandAssignment') || ...
                    strcmp(obj.OutputMode,'PassStopAssignment') );
            else
                flag = false;
            end
        end        
        
        function N = getNumInputsImpl(obj)
            if strcmp(obj.OutputMode,'RegionOfSupport') || ...
                    strcmp(obj.OutputMode,'SubbandAssignment') 
                N = 1;
            elseif strcmp(obj.OutputMode,'AmplitudeSpecification')
                N = 3;
            else
                N = 2;
            end
        end
        
        function N = getNumOutputsImpl(obj)
            if strcmp(obj.OutputMode,'PassStopAssignment') || ...
                    strcmp(obj.OutputMode,'AmplitudeSpecification')
                N = 2;
            else
                N = 1;
            end
        end
        
%         function setupImpl(obj,input)
%         end
        
        function varargout = stepImpl(obj,varargin)
            update_(obj);
            if strcmp(obj.OutputMode,'RegionOfSupport')
                nPoints = varargin{1};
                output  = getRegionOfSupport_(obj,nPoints);
                varargout{1} = output;
            elseif strcmp(obj.OutputMode,'SubbandLocation')
                nPoints = varargin{1};
                idx     = varargin{2};
                if strcmp(obj.SplitMode,'QuarterSplit')
                    output = getSubbandLocationQuarterSplit_(obj,nPoints,idx);
                else
                    output = getSubbandLocation_(obj,nPoints,idx);
                end
                varargout{1} = logical(output);
            elseif strcmp(obj.OutputMode,'PassStopAssignment')
                nPoints = varargin{1};
                idx     = varargin{2};
                if strcmp(obj.SplitMode,'QuarterSplit')
                    [value,sbIdx] = getPassStopAssignmentQuarterSplit_(obj,nPoints,idx);
                else
                    [value,sbIdx] = getPassStopAssignment_(obj,nPoints,idx);
                end                
                varargout{1} = value;
                varargout{2} = sbIdx;
            elseif strcmp(obj.OutputMode,'AmplitudeSpecification')
                nPoints = varargin{1};
                idx     = varargin{2};
                nZeros  = varargin{3};
                if isempty(nZeros)
                    nZeros = [ 2 2 ];
                end
                [value,sbIdx] = ...
                    getAmplitudeSpecification_(obj,nPoints,idx,nZeros);
                varargout{1} = value;
                varargout{2} = sbIdx;
            elseif strcmp(obj.OutputMode,'SubbandAssignment')
                nPoints = varargin{1};
                if strcmp(obj.SplitMode,'QuarterSplit')
                    output  = getSubbandAssignmentQuarterSplit_(obj,nPoints);
                else
                    output  = getSubbandAssignment_(obj,nPoints);
                end                
                varargout{1} = output;
            else
                varargout{1} = [];
            end
            
        end
        
    end
    
    methods ( Access = private)

        function value = getSubbandLocationQuarterSplit_(obj,nPoints,idx)
            VERTICAL   = saivdr.dictionary.utility.Direction.VERTICAL;
            HORIZONTAL = saivdr.dictionary.utility.Direction.HORIZONTAL;
            My = obj.DecimationFactor(VERTICAL);
            Mx = obj.DecimationFactor(HORIZONTAL);
            modx = floor((idx-1)/My);
            mody = mod((idx-1),My);
            if (mody == 0 && modx == 0) || ...
                    (mody == My/2 && modx == 0) || ...
                    (mody == 0 && modx == Mx/2 ) || ...
                    (mody == My/2 && modx == Mx/2)
                value = getRegionOfSupport_(obj,nPoints);
                shiftsize = floor((nPoints./obj.DecimationFactor).*[mody modx]);
                value = circshift(value,shiftsize);
            elseif (mody == 0 ) || (mody == My/2)
                half1 = getHalfRegionOfSupport_(obj,nPoints,'positive',...
                    VERTICAL);
                half2 = getHalfRegionOfSupport_(obj,nPoints,'negative',...
                    VERTICAL);
                shiftsize = floor((nPoints./obj.DecimationFactor).*[mody modx]);
                half1 = circshift(half1,shiftsize);
                half2 = circshift(half2,-shiftsize);
                value = half1 | half2;
            elseif (modx == 0 ) || (modx == Mx/2)
                half1 = getHalfRegionOfSupport_(obj,nPoints,'positive',...
                    HORIZONTAL);
                half2 = getHalfRegionOfSupport_(obj,nPoints,'negative',...
                    HORIZONTAL);
                shiftsize = floor((nPoints./obj.DecimationFactor).*[mody modx]);
                half1 = circshift(half1,shiftsize);
                half2 = circshift(half2,-shiftsize);
                value = half1 | half2;
            else
                qtl = getQuarterRegionOfSupport_(obj,nPoints,'tl');
                qtr = getQuarterRegionOfSupport_(obj,nPoints,'tr');
                qbl = getQuarterRegionOfSupport_(obj,nPoints,'bl');
                qbr = getQuarterRegionOfSupport_(obj,nPoints,'br');
                shiftsize = floor((nPoints./obj.DecimationFactor).*[mody modx]);
                qtl = circshift(qtl,[-1 -1].*shiftsize);
                qtr = circshift(qtr,[-1  1].*shiftsize);
                qbl = circshift(qbl,[ 1 -1].*shiftsize);
                qbr = circshift(qbr,[ 1  1].*shiftsize);
                value = qtl | qtr | qbl | qbr ;
            end
        end
         
         function value = getSubbandIndices_(obj)
             nSubbands = prod(obj.DecimationFactor);
             decY = obj.DecimationFactor(saivdr.dictionary.utility.Direction.VERTICAL);
             eo = zeros(1,nSubbands);
             for idx = 1:nSubbands
                 if idx <= round(nSubbands/2)
                     if mod(idx-1,decY)+1 > round(decY/2)
                         eo(idx) = 1;
                     end
                 else
                     if mod(idx-1,decY)+1 <= round(decY/2)
                         eo(idx) = 1;
                     end
                 end
             end
             aidx = find(eo==1);
             halen = length(aidx)/2;
             sbord = [ find(eo==0) aidx(halen+1:end) aidx(1:halen) ];
             value = zeros(1,nSubbands);
             for idx = 1:nSubbands
                 value(idx) = find(sbord == idx);
             end
         end
         
         function value = getSubbandAssignment_(obj,nPoints)
             value = getSubbandLocation_(obj,nPoints,1);
             for idx = 2:prod(obj.DecimationFactor)
                 value = value + ...
                     idx * getSubbandLocation_(obj,nPoints,idx);
             end
         end
         
         function value = getSubbandAssignmentQuarterSplit_(obj,nPoints)
             value = getSubbandLocationQuarterSplit_(obj,nPoints,1);
             for idx = 2:prod(obj.DecimationFactor)
                 value = value + ...
                     idx * getSubbandLocationQuarterSplit_(obj,nPoints,idx);
             end
         end
         
         function [value,sbIdx] = ...
                 getAmplitudeSpecification_(obj,nPoints,subband,nZeros)
             VERTICAL = saivdr.dictionary.utility.Direction.VERTICAL;
             a = @(x) obj.maxflatfreq(x,nZeros(1),nZeros(2));
 
             if obj.Direction==VERTICAL
                 b = @(x,y) 2*a(x).*a(-obj.Alpha*x+y); % d=y
             else
                 b = @(x,y) 2*a(x-obj.Alpha*y).*a(y); % d=x
             end
             [x,y] = meshgrid(-pi:2*pi/nPoints(2):pi-2*pi/nPoints(2),...
                 -pi:2*pi/nPoints(1):pi-2*pi/nPoints(1));
             if subband == 2
                 value = b(x,y+pi);
             elseif subband == 3
                 value = b(x+pi,y);
             elseif subband == 4
                 value = b(x+pi,y+pi);
             else
                 value = b(x,y);
             end
             if nargout > 1
                 sbIdx = obj.subbandIndices(subband);
             end
         end
         
         function [value,sbIdx] = ...
                 getPassStopAssignment_(obj,nPoints,subband)
             if nargin < 3
                 subband = 1;
             end
             passBand = getSubbandLocation_(obj,nPoints,subband);
             obj.Transition = -obj.Transition;
             stopBand = getSubbandLocation_(obj,nPoints,subband);
             obj.Transition = -obj.Transition;
             value = passBand + stopBand - 1;
             sbIdx = obj.subbandIndices(subband);
         end
         
         function [value, sbIdx] = ...
                 getPassStopAssignmentQuarterSplit_(obj,nPoints,subband)
             passBand = getSubbandLocationQuarterSplit_(obj,nPoints,subband);
             obj.Transition = -obj.Transition;
             stopBand = getSubbandLocationQuarterSplit_(obj,nPoints,subband);
             obj.Transition = -obj.Transition;
             value = passBand + stopBand - 1;
             sbIdx = obj.subbandIndices(subband);
         end
         
         function value = getRegionOfSupport_(obj,nPoints)
             scale = 1 - obj.Transition;
             rosspec = [-scale scale -scale scale];
             value = calcRos_(obj,rosspec,nPoints);
         end

         function obj = update_(obj)
             if obj.Direction == ...
                     saivdr.dictionary.utility.Direction.VERTICAL
                 obj.matrixE(2,1) = - obj.Alpha;
                 obj.matrixE(1,2) = 0;
             else
                 obj.matrixE(1,2) = - obj.Alpha;
                 obj.matrixE(2,1) = 0;
             end
             obj.matrixInvV = (obj.matrixD * obj.matrixE).';
             obj.matrixV = inv(obj.matrixInvV);
         end
            
         function value = ...
                 getHalfRegionOfSupport_(obj,nPoints,region,direction)
             VERTICAL   = saivdr.dictionary.utility.Direction.VERTICAL;
             HORIZONTAL = saivdr.dictionary.utility.Direction.HORIZONTAL;
             scale = 1 - obj.Transition;
             if nargin < 4
                 direction = obj.Direction;
             elseif strcmp(direction,'opposit') && ...
                     obj.Direction == HORIZONTAL
                 direction = VERTICAL;
             elseif strcmp(direction,'opposit') && ...
                     obj.Direction == VERTICAL
                 direction = HORIZONTAL;
             end
             if strcmp(region,'positive')
                 if direction == VERTICAL
                     rostop = -scale;
                     rosbottom = scale;
                     rosleft = obj.Transition;
                     rosright = scale;
                 else
                     rostop = obj.Transition;
                     rosbottom = scale;
                     rosleft = -scale;
                     rosright = scale;
                 end
             elseif strcmp(region,'negative')
                 if direction == VERTICAL
                     rostop = -scale;
                     rosbottom = scale;
                     rosleft = -scale;
                     rosright = -obj.Transition;
                 else
                     rostop = -scale;
                     rosbottom = -obj.Transition;
                     rosleft = -scale;
                     rosright = scale;
                 end
             end
             rosspec = [rostop rosbottom rosleft rosright];
             value = calcRos_(obj,rosspec,nPoints);
         end
         
         function value = getQuarterRegionOfSupport_(obj,nPoints,position)
             scale = 1 - obj.Transition;
             if strcmp(position,'tl')
                 rostop = -scale;
                 rosbottom = -obj.Transition;
                 rosleft = -scale;
                 rosright = -obj.Transition;
             elseif strcmp(position,'tr')
                 rostop = -scale;
                 rosbottom = -obj.Transition;
                 rosleft = obj.Transition;
                 rosright = scale;
             elseif strcmp(position,'bl')
                 rostop = obj.Transition;
                 rosbottom = scale;
                 rosleft = -scale;
                 rosright = -obj.Transition;
             elseif strcmp(position,'br')
                 rostop = obj.Transition;
                 rosbottom = scale;
                 rosleft = obj.Transition;
                 rosright = scale;
             else
                 error('Invalid position');
             end
             rosspec = [rostop rosbottom rosleft rosright];
             value = calcRos_(obj,rosspec,nPoints);
         end
         
         function value = calcRos_(obj,rosspec,nPoints)
             value = zeros(nPoints);
             nRows = nPoints(1);
             nCols = nPoints(2);
             for iCol=1:nCols
                 y(2) = (2.0*(iCol-1)/nCols-1.0); % from 0.0 to 2.0
                 for iRow=1:nRows
                     y(1) = (2.0*(iRow-1)/nRows-1.0); % from 0.0 to 2.0
                     x = obj.matrixInvV*y(:);
                     intX = 4*floor(x/4+0.5);
                     fracX = x - intX;
                     periodInY = mod(obj.matrixV*intX,2);
                     if periodInY(1) == 0 && periodInY(2) == 0
                         if fracX(1) >= rosspec(1) && ...
                                 fracX(1) < rosspec(2)  &&...
                                 fracX(2) >= rosspec(3) && ...
                                 fracX(2) < rosspec(4)
                             value(iRow,iCol) = 1;
                         end
                     end
                 end
             end
         end
       
         function value = getSubbandLocation_(obj,nPoints,idx)
             VERTICAL   = saivdr.dictionary.utility.Direction.VERTICAL;
             HORIZONTAL = saivdr.dictionary.utility.Direction.HORIZONTAL;
             modx = floor((idx-1)/obj.DecimationFactor(VERTICAL));
             mody = mod((idx-1),obj.DecimationFactor(VERTICAL));
             if (mody == 0 && modx == 0) || ...
                     (mody == obj.DecimationFactor(VERTICAL)/2 && modx == 0) || ...
                     (mody == 0 && modx == obj.DecimationFactor(HORIZONTAL)/2 ) || ...
                     (mody == obj.DecimationFactor(VERTICAL)/2 && ...
                     modx == obj.DecimationFactor(HORIZONTAL)/2)
                 value = getRegionOfSupport_(obj,nPoints);
                 shiftsize = floor((nPoints./(obj.DecimationFactor)).*[mody modx]);
                 value = circshift(value,shiftsize);
             else
                 half1 = getHalfRegionOfSupport_(obj,nPoints,'positive',...
                     'opposit');
                 half2 = getHalfRegionOfSupport_(obj,nPoints,'negative',...
                     'opposit');
                 shiftsize = floor((nPoints./(obj.DecimationFactor)).*[mody modx]);
                 half1 = circshift(half1,shiftsize);
                 half2 = circshift(half2,-shiftsize);
                 value = half1 | half2;
             end
         end
       
    end
    
    methods ( Access=private, Static=true)
         function value = maxflatfreq(w,K,L)
             %
             % w: angular frequency
             % K: # of zeros at pi
             % L: # of zeros at zero
             %
             % Reference: P.P. Vaidyanathan, "On Maximally-Flat Linear-
             % Phase FIR Filters," IEEE Trans. on CAS, Vol.21, No.9, 
             % pp.830-832, Sep. 1984
             %
             value = 0.0;
             for idx=1:L
                 n = idx-1;
                 value = value + nchoosek(K-1+n,n)*sin(w/2).^(2*n);
             end
             value = cos(w/2).^(2*K).*value;
         end
    end
end
