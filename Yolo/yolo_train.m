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
opts.train.numSubBatches = 32 ;
opts.train.continue = true ;
opts.train.learningRate = 1e-2 * [ones(1,75), 0.1*ones(1,30), 0.01*ones(1,30)];
opts.train.numEpochs = 135 ;
opts.train.weightDecay = 0.0005 ;
opts.train.derOutputs = {'yololoss', 1} ;
opts.train.expDir = opts.expDir ;
opts.numFetchThreads = 12 ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%    Network initialization
% -------------------------------------------------------------------------
net = load(opts.modelPath);
net = yolo_net_init(net,opts);
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


% -------------------------------------------------------------------------
%  Train
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
%   Deploy
% -------------------------------------------------------------------------



