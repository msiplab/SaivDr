function fcn_setup_ksvdsbox()
%FCN_SETUP_KSVDSBOX Setup function for ksvdsbox and ompsbox
%
% SVN identifier:
% $Id: fcn_setup_ksvdsbox.m 867 2015-11-24 04:54:56Z sho $
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014, Shogo MURAMATSU
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

%%
if exist('ompsbox/omps.m','file') ~= 2 || ...
        exist('ompsbox/private','dir') ~= 7 
    disp('Downloading and unzipping ompsbox1.zip...')
    url = 'http://www.cs.technion.ac.il/%7Eronrubin/Software/ompsbox1.zip';
    disp(url)
    unzip(url,'./ompsbox')
else
    disp('./ompsbox already exists.')
    fprintf('See %s\n', ...
        'http://www.cs.technion.ac.il/%7Eronrubin/software.html');
end
addpath([ pwd '/ompsbox' ])

%%
cdir = pwd;
cd('./ompsbox/private')
if exist('ompsmex','file') ~= 3
    % Make    
    make
end
cd(cdir)

%%
if exist('ksvdsbox/ksvds.m','file') ~= 2 || ...
        exist('ksvdsbox/private','dir') ~= 7 
    disp('Downloading and unzipping ksvdsbox11.zip...')
    url = 'http://www.cs.technion.ac.il/%7Eronrubin/Software/ksvdsbox11.zip';
    disp(url)    
    unzip(url,'./ksvdsbox')
    disp('Done!')
else
    disp('./ksvdsbox already exists.')
    fprintf('See %s\n', ...
        'http://www.cs.technion.ac.il/%7Eronrubin/software.html');
end
addpath([ pwd '/ksvdsbox' ])

%%
cdir = pwd;
cd('./ksvdsbox/private')
if exist('sprow','file') ~= 3
    % Make    
    make
end
cd(cdir)

