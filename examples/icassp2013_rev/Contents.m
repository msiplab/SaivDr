% ICASSP2013_REV Sample codes for the following reference:
%
% Reference
%   Shogo Muramatsu and Natsuki Aizawa, "Lattice Structures for 2-D 
%   Non-separable Oversampled Lapped Transforms," Proc. of 2013 IEEE Intl. 
%   Conf. on Acoustics, Speech and Signal Process. (ICASSP), pp.5632-5636, 
%   May 2013
%
% Files
%   disp_design_examples - Display design examples
%   disp_sweepresults    - Show sweep results of Lambda vs PSNR/SSIM 
%   main_haarimip        - Image Inpainting with Haar transform
%   main_lmax4imrstr     - Pre-calculation of Lipshitz Constant
%   main_nsoltimip       - Image Inpainting with Type-II NSOLT
%   main_parnsoltdsgn    - NSOLT design process 
%   main_sweeplambdaip   - Sweep lambda for image inpainting
%   main_udhaarimip      - Image Inpainting with Undecimated Haar transform
%   main_udirsowtsimip   - Image Inpainting with Union of DirSOWTs 
%
% * NOTE
% 
% - Seven-channel Type-II NSOLT was redesigned and improved the performance
%   from the original shown in the above article.
% 
% - In the above article, a direct tree construction was adopted 
%   as a non-subsampled Haar wavelet transform (NSHAAR). 
%   In this edition, the transform was properly replaced by 
%   a tree structure with upsampling as shown in the following reference:
%  
%   Michael Unser, "Texture classification and segmentation using
%   wavelet frames," vol. 4, no. 11, pp. 1549, Nov. 1995.
%
% - MSE and PSNR evaluation were revised from the original article
%   so that the original and processed picture are compared in 
%   the unsighed byte (8-bit) precision.
%   Please see saivdr.utilit.StepMonitorSystem.
% 
% * Brief introduction
% 
% - Execute an M-file of which name begins with 'main_', e.g. 
% 
%  >> main_nsoltimip
% 
%   and then execute an M-file of which name begins with 'disp_', e.g. 
%  
%  >> disp_design_examples
% 
% * Contact address: 
% 
%    Shogo MURAMATSU,
%    Faculty of Engineering, Niigata University,
%    8050 2-no-cho Ikarashi, Nishi-ku,
%    Niigata, 950-2181, JAPAN
%    LinkedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
% 
% SVN identifier:
% $Id: Contents.m 287 2014-03-13 08:41:09Z sho $
