classdef AmplitudeErrorEnergyTestCase < matlab.unittest.TestCase
    %AMPLITUDEERRORENERGYTESTCASE Test case for AmplitudeErrorEnergy
    %
    % SVN identifier:
    % $Id: AmplitudeErrorEnergyTestCase.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2015, Shogo MURAMATSU
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
    
    properties
        aee
    end
    
    methods (TestMethodTeardown)
        
        function deleteObject(testCase)
            delete(testCase.aee)
        end

    end
    
    methods (Test)
        
        % Test 
        function testDefaultRand16x16(testCase)
            
            % Parameters
            nPoints = [16 16];

            % Synthesis filters for test 
            spcAmpLyLx = rand(nPoints); 
            spcAmpHyLx = rand(nPoints); 
            spcAmpLyHx = rand(nPoints); 
            spcAmpHyHx = rand(nPoints); 
            
            % Band specification
            import saivdr.dictionary.utility.Subband
            specBand(:,:,Subband.LyLx) = spcAmpLyLx;
            specBand(:,:,Subband.HyLx) = spcAmpHyLx;
            specBand(:,:,Subband.LyHx) = spcAmpLyHx;
            specBand(:,:,Subband.HyHx) = spcAmpHyHx;
            
            % Impulse response
            [x,y] = meshgrid(-pi:2*pi/nPoints(2):pi-2*pi/nPoints(2),...
                -pi:2*pi/nPoints(1):pi-2*pi/nPoints(1));
            halfdelay = exp(-1i*(x+y)/2);
            filtImp(:,:,Subband.LyLx) = circshift(... % TODO: Concrete values
                ifft2(ifftshift(spcAmpLyLx.*halfdelay)),nPoints/2-1);
            filtImp(:,:,Subband.HyLx) = circshift(...
                ifft2(ifftshift(spcAmpHyLx.*halfdelay)),nPoints/2-1);
            filtImp(:,:,Subband.LyHx) = circshift(...
                ifft2(ifftshift(spcAmpLyHx.*halfdelay)),nPoints/2-1);
            filtImp(:,:,Subband.HyHx) = circshift(...
                ifft2(ifftshift(spcAmpHyHx.*halfdelay)),nPoints/2-1);
            
            % Expected values
            energyExpctd = 0.0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.design.*
            testCase.aee = AmplitudeErrorEnergy(...
                'AmplitudeSpecs',specBand);
            
            % Actual values
            energyActual = step(testCase.aee,filtImp);
            
            % Evaluation
            testCase.verifyEqual(energyActual,energyExpctd,'AbsTol',1e-15);
            
        end
        
        % Test for default construction
        function testDefaultRand16x16WithNullBasis(testCase)
            
            % Parameters
            nPoints = [16 16];
            
            % Synthesis filters for test
            spcAmpLyLx = rand(nPoints); 
            spcAmpHyLx = rand(nPoints); 
            spcAmpLyHx = rand(nPoints); 
            spcAmpHyHx = rand(nPoints); 
            
            % Band specification
            import saivdr.dictionary.utility.Subband
            specBand(:,:,Subband.LyLx) = spcAmpLyLx;
            specBand(:,:,Subband.HyLx) = spcAmpHyLx;
            specBand(:,:,Subband.LyHx) = spcAmpLyHx;
            specBand(:,:,Subband.HyHx) = spcAmpHyHx;
            
            % Impulse response
            filtImp(:,:,Subband.LyLx) = zeros(nPoints);
            filtImp(:,:,Subband.HyLx) = zeros(nPoints);
            filtImp(:,:,Subband.LyHx) = zeros(nPoints);
            filtImp(:,:,Subband.HyHx) = zeros(nPoints);
            
            % Expected values
            energyExpctd = ...
                spcAmpLyLx(:).'*spcAmpLyLx(:) + ...
                spcAmpHyLx(:).'*spcAmpHyLx(:) + ...
                spcAmpLyHx(:).'*spcAmpLyHx(:) + ...
                spcAmpHyHx(:).'*spcAmpHyHx(:);
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.design.*
            testCase.aee = AmplitudeErrorEnergy(...
                'AmplitudeSpecs',specBand);
            
            % Actual values
            energyActual = step(testCase.aee,filtImp);
            
            % Evaluation
            testCase.verifyEqual(energyActual,energyExpctd,'AbsTol',1e-15);            
            
        end

        % Test for default construction
        function testDefaultRand64x64(testCase)
            
            % Parameters
            nPoints = [64 64];

            % Synthesis filters for test 
            spcAmpLyLx = rand(nPoints); 
            spcAmpHyLx = rand(nPoints); 
            spcAmpLyHx = rand(nPoints); 
            spcAmpHyHx = rand(nPoints); 
            
            % Band specification
            import saivdr.dictionary.utility.Subband
            specBand(:,:,Subband.LyLx) = spcAmpLyLx;
            specBand(:,:,Subband.HyLx) = spcAmpHyLx;
            specBand(:,:,Subband.LyHx) = spcAmpLyHx;
            specBand(:,:,Subband.HyHx) = spcAmpHyHx;
            
            % Impulse response
            [x,y] = meshgrid(-pi:2*pi/nPoints(2):pi-2*pi/nPoints(2),...
                -pi:2*pi/nPoints(1):pi-2*pi/nPoints(1));
            halfdelay = exp(-1i*(x+y)/2);
            filtImp(:,:,Subband.LyLx) = circshift(... % TODO: Concrete values
                ifft2(ifftshift(spcAmpLyLx.*halfdelay)),nPoints/2-1);
            filtImp(:,:,Subband.HyLx) = circshift(...
                ifft2(ifftshift(spcAmpHyLx.*halfdelay)),nPoints/2-1);
            filtImp(:,:,Subband.LyHx) = circshift(...
                ifft2(ifftshift(spcAmpLyHx.*halfdelay)),nPoints/2-1);
            filtImp(:,:,Subband.HyHx) = circshift(...
                ifft2(ifftshift(spcAmpHyHx.*halfdelay)),nPoints/2-1);
            
            % Expected values
            energyExpctd = 0.0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.design.*
            testCase.aee = AmplitudeErrorEnergy(...
                'AmplitudeSpecs',specBand);
            
            % Actual values
            energyActual = step(testCase.aee,filtImp);
            
            % Evaluation
            testCase.verifyEqual(energyActual,energyExpctd,'AbsTol',1e-15);
            
        end
        
        % Test for default construction
        function testGetCostAt(testCase)
            
            % Parameters
            nPoints = [16 16];
        
            % Synthesis filters for test
            spcAmpLyLx = rand(nPoints); 
            spcAmpHyLx = rand(nPoints); 
            spcAmpLyHx = rand(nPoints); 
            spcAmpHyHx = rand(nPoints); 
            
            % Band specification
            import saivdr.dictionary.utility.Subband
            specBand(:,:,Subband.LyLx) = spcAmpLyLx;
            specBand(:,:,Subband.HyLx) = spcAmpHyLx;
            specBand(:,:,Subband.LyHx) = spcAmpLyHx;
            specBand(:,:,Subband.HyHx) = spcAmpHyHx;
            
            % Impulse response
            filtImp(:,:,Subband.LyLx) = zeros(nPoints);
            filtImp(:,:,Subband.HyLx) = zeros(nPoints);
            filtImp(:,:,Subband.LyHx) = zeros(nPoints);
            filtImp(:,:,Subband.HyHx) = zeros(nPoints);
            
            % Expected values
            energyExpctd = [
                spcAmpLyLx(:).'*spcAmpLyLx(:);
                spcAmpHyLx(:).'*spcAmpHyLx(:);
                spcAmpLyHx(:).'*spcAmpLyHx(:);
                spcAmpHyHx(:).'*spcAmpHyHx(:) ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.design.*
            testCase.aee = AmplitudeErrorEnergy(...
                'AmplitudeSpecs',specBand,...
                'EvaluationMode','Individual');
            
            % Evaluation
            for idx = 1:length(energyExpctd)
                % Actual values
                energyActual = step(testCase.aee,filtImp(:,:,idx),idx);
                testCase.verifyEqual(energyActual,energyExpctd(idx),...
                    'AbsTol',1e-15);
            end
            
        end

    end
end
