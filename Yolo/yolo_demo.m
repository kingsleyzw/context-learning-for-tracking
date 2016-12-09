function yolo_demo(varargin)
%FAST_RCNN_DEMO  Demonstrates Fast-RCNN
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..','externel', 'matconvnet','matlab', 'vl_setupnn.m')) ;
addpath('bbox_functions');
opts.dataDir = fullfile(fileparts(mfilename('fullpath')), '..\..','data');
opts.modelPath = fullfile(opts.dataDir, 'exp', 'net-deployed.mat');
opts.classes = {'person'} ;
opts.gpu = [] ;
opts.confThreshold = 0.1 ;
opts.nmsThreshold = 0.3 ;
opts.width = 448; % a size compatible with the network.
opts. height = 448;
opts = vl_argparse(opts, varargin) ;

para.side = 7;
para.output = 30;

% Load the network and put it in test mode.
net = load(opts.modelPath) ;
net = dagnn.DagNN.loadobj(net);
net.mode = 'test' ;
% Mark  predictions as `precious` so they are not optimized away during evaluation.
net.vars(net.getVarIndex('prediction')).precious = 1 ;
% Load a test image and candidate bounding boxes.
im = single(imread('000001.jpg')) ;
imageSize = size(im) ;
imo = im; % keep original image 
% Resize images and boxes to a size compatible with the network.
im = imresize(im,[opts.height opts. width],'Method',net.meta.normalization.interpolation);
% Remove the average color from the input image.
imNorm = bsxfun(@minus, im, net.meta.normalization.averageImage) ;

% Evaluate network either on CPU or GPU.
if numel(opts.gpu) > 0
  gpuDevice(opts.gpu) ;
  imNorm = gpuArray(imNorm) ;
  net.move('gpu') ;
end
net.conserveMemory = false ;
net.eval({'input', imNorm});
% Extract box coordinates, confidence and class probabilities 
prediction = squeeze(gather(net.vars(net.getVarIndex('prediction')).value)) ;
result = reshape(prediction,para.output,para.side^2);
conf = result(5 , :);
cboxes = bbox_std(result(1:4 ,:) , imageSize);
for i = 1:numel(opts.classes)
  c = find(strcmp(opts.classes{i}, net.meta.classes.name)) ;
  cprobs = result(c+10,:).*conf ;
  cls_dets = [cboxes ; cprobs]' ;
  keep = bbox_nms(cls_dets, opts.nmsThreshold) ;% for Non-maximum suppression
  cls_dets = cls_dets(keep, :) ;
  sel_boxes = find(cls_dets(:,end) >= opts.confThreshold) ;
  imo = bbox_draw(imo/255,cls_dets(sel_boxes,:));
  title(sprintf('Detections for class ''%s''', opts.classes{i})) ;
  fprintf('Detections for category ''%s'':\n', opts.classes{i});
  for j=1:size(sel_boxes,1)
    bbox_id = sel_boxes(j,1);
    fprintf('\t(%.1f,%.1f)\t(%.1f,%.1f)\tprobability=%.6f\n', ...
            cls_dets(bbox_id,1), cls_dets(bbox_id,2), ...
            cls_dets(bbox_id,3), cls_dets(bbox_id,4), ...
            cls_dets(bbox_id,end));
  end
end

