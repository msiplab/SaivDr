% ICIP2016 Sample codes for the following reference:
% 
% This folder contains supplemental materials for the following
% ICIP2016 paper:
%
% Reference
%   Shogo Muramatsu, Masaki Ishii and Zhiyu Chen,
%   "Efficient Parameter Optimization for Example-Based Design of 
%   Non-separable Oversampled Lapped Transform," Proc. of 2016 IEEE Intl.
%   Conf. on Image Process. (ICIP), Sept. 2016, to appear
%
% Files
%
%   main_sec_4_1_table1        - Computational Time of NSOLT Dictionary Update
%   main_sec_4_2_atmimg_nsolt2 - Display Atomic Images of 2-D NSOLT
%   main_sec_4_2_design_nsolt2 - Dictionary Learning with 2-D NSOLT
%   main_sec_4_2_design_sksvd2 - Dictionary Learning with 2-D Sparse K-SVD
%   main_sec_4_2_table3        - Sparse Approximation with 2-D NSOLT
%
% ====================================================================
% 
% a) Description:
% 
%    This folder contains MATLAB script, function and class definition
%    codes so that readers can reproduce the following materials in the
%    paper:
% 
%     * Section 4.1: Numerical versus analytical gradient
% 
%       - Table 1: Sparse approximation results in PSNR
% 
%     * Section 4.2: Design examples with SGD
% 
%       - Fig. 2: Original images for dictionary learning and sparse Aprx.
%       - Fig. 3: Atomic images of a learned NSOLT
%       - Table 3: Sparse approximation results in PSNR
% 
% b) Platform:
% 
%    MATLAB R2015b or later
% 
% c) Environment:
% 
%    The attached MATLAB codes have been tested on the following
%    operating systems.
% 
%     * Windows 8.1 Pro (64bit) 
% 
%    In order to execute the attached codes, the following MATLAB
%    Toolboxes are required.
%    
%     * Signal Processing Toolbox
%     * Image Processing Toolbox
% 
%    In addition, the codes for the design processes require the
%    following options.
% 
%     * Optimization Toolbox
%     
%    It is also recommended to prepare the following Toolboxes for
%    accelerating the execution.
% 
%     * MATLAB Coder
%     * Parallel Computing Toolbox
% 
%    Some of the MATLAB codes try to download images and additional
%    program codes from the Internet. Unless necessary materials are not 
%    downloaded, those codes must be executed with the Internet connection. 
% 
% d) Major Component Description:
% 
%    The MATLAB scripts of which name begin with 'main_' under the 
%    top folder reproduce the results shown in the paper.
% 
%     * main_sec_4_1_table1 - Computational Time of NSOLT Dictionary Update (Table 1)
%     * main_sec_4_2_design_sksvd2 - Dictionary Learning with 2-D Sparse K-SVD
%     * main_sec_4_2_design_nsolt2 - Dictionary Learning with 2-D NSOLT
%     * main_sec_4_2_atmimg_nsolt2 - Display Atomic Images of 2-D NSOLT (Fig. 3)
%     * main_sec_4_2_table3 - Spare Approximation with 2-D NSOLT (Fig. 2,Table 3)
% 
%    Folder 'examples/+support' contains some functions called 
%    by the above scripts during their execution.
% 
%    Under folder '+saivdr,' the class definitions distributed as the
%    SaivDr package are found. Files 'Contents.m' and 'RELEASENOTE.txt'
%    on the top layer serve some information on the package.
% 
% e) Set-up Instructions:
% 
%    Move to the top layer of the extracted folder, and type 'setpath' 
%    at the command window as
%
%     >> setpath
% 
%    Then, move to examples/icip2016
%
%     >> cd examples/icip2016
%
% f) Run Instructions:
% 
%    Just input a main script name which begins by 'main_' on the MATLAB
%    command window. For example,
%  
%     >> main_sec_4_1_table1
% 
%    If MATLAB Coder is available, some scripts try to generate
%    executable codes as MEX-files at the first time. Once the code
%    generation successfully finishes, such MEX-files would be stored in
%    folder 'mexcodes' under the top layer and the execution would
%    become faster.
% 
% h) Output Description:
% 
%    The following scripts produced the design examples shown in 
%    Section 4.2. Folder 'results' under the top layer contains
%    the resulting design parameter values. For the detail, please see 
%    the header comments on the following scripts:
% 
%     * main_sec_4_2_design_sksvd2
%     * main_sec_4_2_design_nsolt2
% 
%    The following scripts store impluse responses of design examples in the
%    folder 'tiff' under the top layer. For the detail, please see the
%    header comments on each script.
% 
%     * main_sec_4_2_atmimg_nsolt2
% 
%    The following scripts generate and store pictures and tables in
%    folders 'tiff' and 'materials' under the top layer,
%    respectively. For the detail, please see the header comments on
%    each script.
% 
%     * main_sec_4_2_table3
% 
% i) Contact Information:
%  
%    Shogo MURAMATSU, Associate Professor
% 
%    Dept. of Electrical and Electronic Eng.,
%    Faculty of Engineering, Niigata University,
%    8050 2-no-cho Ikarashi, Nishi-ku, Niigata, 950-2181, JAPAN
% 
%    Email: shogo@eng.niigata-u.ac.jp
%    	  shogo.muramatsu.jp@ieee.org
% 
% ---





