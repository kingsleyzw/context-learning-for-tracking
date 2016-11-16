%% initialise the tracker

% load videos
v{1}=VideoReader('../CLTdataset/dataset3-4cameras/Cam1.avi');
v{2}=VideoReader('../CLTdataset/dataset3-4cameras/Cam2.avi');
v{3}=VideoReader('../CLTdataset/dataset3-4cameras/Cam3.avi');
v{4}=VideoReader('../CLTdataset/dataset3-4cameras/Cam4.avi');

% load annotations
annotation_path='../CLTdataset/annotation_files/annotation/Dataset3/at least 4/';
annotation_files = dir([annotation_path '*.mat']);

%load cameras' detection
cam{1}  = load('../CLTdataset/annotation_files/annotation/Dataset3/Cam1.dat');
cam{2}  = load('../CLTdataset/annotation_files/annotation/Dataset3/Cam2.dat');
cam{3}  = load('../CLTdataset/annotation_files/annotation/Dataset3/Cam3.dat');
cam{4}  = load('../CLTdataset/annotation_files/annotation/Dataset3/Cam4.dat');

org_det  = [cam{1}; cam{2}; cam{3}; cam{4}]; % sort out sequential orders
[~, idd] = sort(org_det(:, 2)); 
det      = org_det(idd, :);
numP     = max(org_det(:, 3)); % number of pedestrian

% load fast VGG
vl_setupnn()
opts.modelPath = fullfile(fileparts(mfilename('fullpath')), ...
'../..', 'models', 'fast-rcnn-vgg16-pascal07-dagnn.mat') ;
opts.gpu = [] ;
net = load(opts.modelPath) ;
net = dagnn.DagNN.loadobj(net);
net.mode = 'test' ; 

% init parameters
flag_single = 0; % 0 : multi-cameras; 1 : single-camera