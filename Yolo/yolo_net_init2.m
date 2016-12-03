function net = yolo_net_init2(varargin)
% -------------------------------------------------------------------------
opts.modelPath = fullfile('data', 'models');
opts = vl_argparse(opts, varargin) ;

net = load(opts.modelPath);
net = vl_simplenn_tidy(net);
