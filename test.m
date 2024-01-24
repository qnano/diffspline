function [PSFsmooth,PSF,zstack,shiftedzstack,shifts]=test_smap(input_filename, output_filename, depths, n_realizations)
%% add path to helper functions
addpath('matlab data generation\calib\');
addpath('matlab data generation\shared\');

%% make bead calibration for 3D astigmatic PSF
%datapath=[pwd filesep 'example_data' filesep 'beadstacks_3D_astig'];
%p.filelist={[datapath filesep 'stack3D_1.tif'],[datapath filesep 'stack3D_2.tif'],[datapath filesep 'stack3D_3.tif']};  % list of image stacks
%p.outputfile = [pwd filesep 'example_data' filesep 'bead_astig_3dcal.mat' ]; %output file name


p.filtersize = 2; %size of the filter (pixels) for bead segmentation. Use larger value for bi-lobed PSF.
p.mindistance = 10; %minimum distance (pixels) between beads
p.dz = 10; %distance between frames in nm
p.zcorr ='cross-correlation'; %modality of z-alignment
p.zcorrframes = 50; % number of frames used for 3D correlation
p.ROIxy = 34; %size of the PSF model in x,y (pixels)
p.smoothz = 1; %smoothing parameter in Z

%p.yrange=[-500 500]; 
%p.gaussrange =[-700 700]; %z range (nm) in which to calibrate the parameters for Gaussian z-fit.
%p.gaussroi =19; %size of the ROI in pixels for the Gaussian calibration

%% make bead calibration for unmodified PSF
%as before, but using the bead stack with an unmodified PSF
%datapath=[pwd filesep 'example_data' filesep 'beadstacks_2D'];
%p.filelist={[datapath filesep 'stack2D_1.tif'],[datapath filesep 'stack2D_2.tif'],[datapath filesep 'stack2D_3.tif']};  % list of image stacks
%input_filename = 'zstack_9beads_100000intensity_20bg.tif'
%output_filename = 'zstack_9beads_100000intensity_20bg_3Dcorr.mat'
%for filename = input_filename

p.filelist=input_filename; %{[pwd filesep input_filename]};
p.outputfile = [pwd filesep output_filename ]; %output file name
p.depths = cell2mat(depths);
p.modality = 'arbitrary'; %'2D PSF'; 
p.n_realizations = n_realizations;
[PSFsmooth,PSF,zstack,shiftedzstack,shifts]=calibrate3D(p); %call calibraion function

end