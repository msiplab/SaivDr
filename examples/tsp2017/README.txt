This folder contains excerpts from and modifications to the following code groups

(Reference site)
https://sigport.org/documents/multidimensional-nonseparable-oversampled-lapped-transforms-theory-and-design

(Original README)
====================================================================
README file for MdNsolt.zip 

This ZIP file contains supplemental materials for paper
"Multidimensional Nonseparable Oversampled Lapped Transforms: Theory
and Design," written by S. MURAMATSU, Kosuke FURUYA and Naotaka YUKI.
====================================================================

a) Description:

   This ZIP file contains MATLAB script, function and class definition
   codes so that readers can reproduce the following materials in the
   paper:

    * Section V.A: 2-D Design Examples

      - Fig. 6: Training image for 2-D NSOLT design
      - Fig. 7: Sparse approximation results with 2-D NSOLT
      - Fig. 8: Impluse responses of 2-D NSOLT synthesizer
      - Table IV: Sparse approximation results in PSNR

    * Section V.B: 3-D Design Examples

      - Fig. 9:  Training volume data for 3-D NSOLT dictionary learning
      - Fig. 10: Impluse responses of 3-D NSOLT synthesizer
      - Table V: Sparse approximation results in PSNR
 
   Note that these materials utilize a subset of SaivDr Package for
   MATLAB, which is developed by the authors and the full latest
   version is available from the following site:

    http://www.mathworks.com/matlabcentral/fileexchange/authors/82558

b) Size:

   838 kB

c) Platform:

   MATLAB R2014a or later

d) Environment:

   The attached MATLAB codes have been tested on the following
   operating systems.

    * Windows 7 Professional SP1 (64bit)
    * Windows 8.1 Pro (64bit) 
    * Red Hat Enterprise Linux Server Release 6.3 (Santiago) (64bit)

   In order to execute the attached codes, the following MATLAB
   Toolboxes are required.
   
    * Signal Processing Toolbox
    * Image Processing Toolbox

   In addition, the codes for the design processes require the
   following options.

    * Optimization Toolbox
    
   It is also recommended to prepare the following Toolboxes for
   accelerating the execution.

    * MATLAB Coder
    * Parallel Computing Toolbox

   Some of the MATLAB codes try to download images, volume data and
   additional program codes from the Internet. Unless necessary
   materials are not downloaded, those codes must be executed with the
   Internet connection. 

e) Major Component Description:

   The MATLAB scripts of which name begin with 'main_' in the top
   layer reproduce the dictionary learning results, figures and
   tables used in the paper.

   Design process:

    * main_sec_v_a_design - 2-D NSOLT dictionary learning (Fig. 6)   
    * main_sec_v_b_design - 3-D NSOLT dictionary learning (Fig. 9)

   Impluse responses:

    * main_sec_v_a_atmimg - Impluse responses of 2-D NSOLT synthesizer (Fig. 8)
    * main_sec_v_b_atmimg - Impluse responses of 3-D NSOLT synthesizer (Fig. 10)

   Sparse approximation:

    * main_sec_v_a_imaprx  - Sparse approximation with 2-D NSOLT (Fig. 7)
    * main_sec_v_a_psnrs   - Sparse approximation results (Table IV)
    * main_sec_v_b_psnrs   - Sparse approximation results (Table V)

   Folders '+eznsolt' and '+support' contain the functions called by
   the above scripts during their execution.

   Under folder '+saivdr,' the class definitions distributed as the
   SaivDr package are found. Files 'Contents.m' and 'RELEASENOTE.txt'
   on the top layer serve some information on the package.

f) Detailed Set-up Instructions:

   Unzip the downloaded file 'MdNsolt.zip.' For example, type the
   following command on the MATLAB command window.

    >> unzip MdNsolt.zip

   Then, move to the top layer of the extracted folder as

    >> cd MdNsolt

g) Detailed Run Instructions:

   Just input a main script name which begins by 'main_' on the MATLAB
   command window. For example,
 
    >> main_sec_v_a_design 

   If MATLAB Coder is available, some scripts try to generate
   executable codes as MEX-files at the first time. Once the code
   generation successfully finishes, such MEX-files would be stored in
   folder 'mexcodes' under the top layer and the execution would
   become faster.

h) Output Description:

   The following scripts produced the design examples shown in Section V.
   Folder 'results' under the top layer contains the resulting
   design parameter values. For the detail, please see the header
   comments on the following scripts:

    * main_sec_v_a_design       
    * main_sec_v_b_design

   The following scripts store impluse responses of design examples in the
   folder 'tiff' under the top layer. For the detail, please see the
   header comments on each script.

    * main_sec_v_a_atmimg
    * main_sec_v_b_atmimg

   The following scripts generate and store pictures and tables in
   folders 'tiff' and 'materials' under the top layer,
   respectively. For the detail, please see the header comments on
   each script.

    * main_sec_v_a_imaprx
    * main_sec_v_a_psnrs 
    * main_sec_v_b_psnrs  

i) Contact Information:
 
   Shogo MURAMATSU, Associate Professor

   Dept. of Electrical and Electronic Eng.,
   Faculty of Engineering, Niigata University,
   8050 2-no-cho Ikarashi, Nishi-ku, Niigata, 950-2181, JAPAN

   Email: shogo@eng.niigata-u.ac.jp
   	  shogo.muramatsu.jp@ieee.org

---
$Id: README.txt 857 2015-11-02 02:21:15Z sho $
