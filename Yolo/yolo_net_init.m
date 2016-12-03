function net = yolo_net_init(varargin)
% -------------------------------------------------------------------------
opts.modelPath = fullfile('data', 'models');
opts = vl_argparse(opts, varargin) ;

net = load(opts.modelPath);
net = vl_simplenn_tidy(net);
%Add 2 pooling layers
pool2p = find(cellfun(@(a) strcmp(a.name, 'pool2'), net.layers)==1);
pool3 = net.layers{pool2p};
pool3.name = 'pool3';
pool4 = net.layers{pool2p};
pool4.name = 'pool4';
relu3p = find(cellfun(@(a) strcmp(a.name, 'relu3'), net.layers)==1);
relu4p = find(cellfun(@(a) strcmp(a.name, 'relu4'), net.layers)==1);
net.layers = [net.layers(1:relu3p) pool3 net.layers(relu3p+1:relu4p) pool4 net.layers(relu4p+1:end)];

% Skip layers from fc6
fc6p = cellfun(@(a) strcmp(a.name, 'fc6'), net.layers)==1;
fc6 = net.layers(fc6p);
conv5p = find(cellfun(@(a) strcmp(a.name, 'conv5'), net.layers)==1);
relu5p = find(cellfun(@(a) strcmp(a.name, 'relu5'), net.layers)==1);
net.layers = net.layers(1:relu5p);
% Add 4 conv layers and 2 fully connected layers as in yolo paper
sizeW{1} = size(net.layers{conv5p}.weights{1});
sizeW{2} = size(net.layers{conv5p}.weights{2});
for i=1:4
    net.layers(relu5p+(i-1)*2+1)=net.layers(conv5p);
    net.layers{relu5p+(i-1)*2+1}.name=['conn_add' num2str(i)];
    net.layers{relu5p+(i-1)*2+1}.weights={0.001 * randn(sizeW{1},'single'), zeros(sizeW{2}, 'single')};
    net.layers(relu5p+(i-1)*2+2)=net.layers(relu5p);
    net.layers{relu5p+(i-1)*2+2}.name=['relu_add' num2str(i)];
end
net.layers(relu5p+9)=fc6;
net.layers{relu5p+9}.name='fc_add1';
net.layers{relu5p+9}.weights={0.001 * randn([6 6 256 4096],'single'), zeros([4096 1], 'single')};
net.layers{relu5p+9}.size=[6 6 256 4096];
net.layers(relu5p+10)=net.layers(relu5p);
net.layers{relu5p+10}.name='relu_add5';
net.layers{relu5p+11}=struct('type', 'dropout', 'rate', 0.5, 'name','drop1');

% Convert to DagNN.
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

% Add prediction layer
pdrop1 = (arrayfun(@(a) strcmp(a.name, 'drop1'), net.layers)==1);
net.addLayer('pred', dagnn.Conv('size',[1 1 4096 1470],'hasBias', true), ...
net.layers(pdrop1).outputs{1}, 'prediction',{'predf','predb'}) ;

net.params(end-1).value = 0.001 * randn(1,1,4096,1470,'single');
net.params(end).value = zeros(1470,1,'single');

% Add yolo loss layer--how to design
net.addLayer('loss', dagnn.yoloLoss(), ...
{'prediction','weight','data'}, 'yololoss',{}) ;
% No decay for bias and set learning rate to 2
for i=2:2:numel(net.params)
  net.params(i).weightDecay = 0;
  net.params(i).learningRate = 2;
end

% Change image-mean
net.meta.normalization.averageImage = ...
  reshape([122.7717 102.9801 115.9465],[1 1 3]);

net.meta.normalization.interpolation = 'bilinear';

net.meta.classes.name = {'aeroplane', 'bicycle', 'bird', ...
    'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', ...
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', ...
    'sofa', 'train', 'tvmonitor', 'background' };






