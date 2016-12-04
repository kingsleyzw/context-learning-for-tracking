function [net, info] = yolo_train(varargin)
% yolo_train fine-tunes a pre-trained CNN on imagenet dataset


run(fullfile(fileparts(mfilename('fullpath')), ...
  '..','externel', 'matconvnet','matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile(fileparts(mfilename('fullpath')), '..\..','data');
opts.modelPath = fullfile(opts.dataDir, 'models', 'imagenet-vgg-f.mat');
opts.expDir  = fullfile(opts.dataDir, 'exp') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.train = struct() ;
opts.train.gpus = [];
opts.train.batchSize = 64 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.learningRate = 1e-2 * [ones(1,75), 0.1*ones(1,30), 0.01*ones(1,30)];
opts.train.numEpochs = 135 ;
opts.train.weightDecay = 0.0005 ;
opts.train.derOutputs = {'yololoss', 1} ;
opts.train.expDir = opts.expDir ;
opts.numFetchThreads = 2 ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%    Network initialization
% -------------------------------------------------------------------------
net = yolo_net_init('modelPath',opts.modelPath);
% net = yolo_net_init2('modelPath',opts.modelPath);
% -------------------------------------------------------------------------
%   Database initialization
% -------------------------------------------------------------------------
if exist(opts.imdbPath,'file')
  fprintf('Loading imdb...');
  imdb = load(opts.imdbPath) ;
else
  if ~exist(opts.expDir,'dir')
    mkdir(opts.expDir);
  end
  imdb = yolo_setup_data('dataDir', opts.dataDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
fprintf('done\n');
% --------------------------------------------------------------------
% Train
% --------------------------------------------------------------------
% use train + val split to train
imdb.images.set(imdb.images.set == 2) = 1;
% minibatch options
bopts = net.meta.normalization;
bopts.useGpu = numel(opts.train.gpus) >  0 ;
bopts.numThreads = opts.numFetchThreads;
bopts.interpolation = net.meta.normalization.interpolation;
bopts.width = 448;
bopts. height = 448;

[net,info] = cnn_train_dag(net, imdb, @(i,b) ...
                           getBatch(bopts,i,b), ...
                           opts.train) ;

% --------------------------------------------------------------------
% Deploy
% --------------------------------------------------------------------
modelPath = fullfile(opts.expDir, 'net-deployed.mat');
if ~exist(modelPath,'file')
  net = yolo_deploy(net);
  net_ = net.saveobj() ;
  save(modelPath, '-struct', 'net_') ;
  clear net_ ;
end

% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
ims = vl_imreadjpeg(images,'numThreads',opts.numThreads) ;
im  = zeros(opts.height, opts.weight , size(ims{1},3),numel(batch),'single');

s = size(imdb.traindata.weight{1},3);
weights = zeros(1,1,s,numel(batch));
data =  zeros(1,1,s,numel(batch));
for b=1:numel(batch)
    ims{b} = imresize(ims{b},[opts.height opts.width],'Method',opts.interpolation);
%     if ~isempty(opts.averageImage)
%         ims{b} = single(bsxfun(@minus,ims{b},opts.averageImage));       
%     end
    im(:,:,:,b) = single(ims{b});   
    weights(:,:,:,b) = imdb.traindata.weight{batch(b)};   
    data(:,:,:,b) = imdb.traindata.data{batch(b)};
end

if opts.useGpu > 0
  im = gpuArray(im) ;
  weights = gpuArray(weights) ;
  data = gpuArray(data) ;
end
inputs = {'input', im, 'weight', weights, 'data', data} ;

% --------------------------------------------------------------------
function net = yolo_deploy(net)
% --------------------------------------------------------------------
for l = numel(net.layers):-1:1
  if isa(net.layers(l).block, 'dagnn.yoloLoss') || ...
      isa(net.layers(l).block, 'dagnn.DropOut')
    layer = net.layers(l);
    net.removeLayer(layer.name);
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
  end
end
net.rebuild();
net.mode = 'test' ;










