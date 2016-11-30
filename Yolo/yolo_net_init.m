function net = yolo_net_init(varargin)
% -------------------------------------------------------------------------
opts.modelPath = fullfile('data', 'models');
opts = vl_argparse(opts, varargin) ;

net = load(opts.modelPath);
net = vl_simplenn_tidy(net);
% Skip layers from fc6
fc6p = cellfun(@(a) strcmp(a.name, 'fc6'), net.layers)==1;
fc6=net.layers(fc6p);
conv5p = find(cellfun(@(a) strcmp(a.name, 'conv5'), net.layers)==1);
relu5p = find(cellfun(@(a) strcmp(a.name, 'relu5'), net.layers)==1);
net.layers = net.layers(1:relu5p);
% Add 4 conv layers as in yolo paper
sizeW{1} = size(net.layers{conv5p}.weights{1});
sizeW{2} = size(net.layers{conv5p}.weights{2});
for i=1:4
    net.layers(relu5p+(i-1)*2+1)=net.layers(conv5p);
    net.layers{relu5p+(i-1)*2+1}.name=['conn_add' num2str(i)];
    net.layers{relu5p+(i-1)*2+1}.weights={0.001 * randn(sizeW{1},'single'), zeros(sizeW{2}, 'single')};
    net.layers(relu5p+(i-1)*2+2)=net.layers(relu5p);
    net.layers{relu5p+(i-1)*2+2}.name=['relu_add' num2str(i)];
end
% Add 2 fully connected layers as in yolo paper
net.layers(relu5p+9)=fc6;
net.layers{relu5p+9}.name='fc_add1';
net.layers{relu5p+9}.weights={0.001 * randn([3 3 256 4096],'single'), zeros([4096 1], 'single')};
net.layers{relu5p+9}.size=[3 3 256 4096];
net.layers(relu5p+10)=fc6;
net.layers{relu5p+10}.name='fc_add2';
net.layers{relu5p+10}.weights={0.001 * randn([1 1 4096],'single'), zeros([4096 1], 'single')};
net.layers{relu5p+10}.size=[1 1 4096];
% Convert to DagNN.
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

% Add yolo loss layer--how to design
net.addLayer('loss', dagnn.yoloLoss(), ...
{'prediction','label'}, {'losscoord','losscls'},{}) ;
