function f = get_features(frame, detections, net)
%% extract the CNN features (using VGG model)

% Resize images and detections to be compatible with the network.
frame = single(frame);
imageSize = size(frame) ;
fullImageSize = net.meta.normalization.imageSize(1) ...
    / net.meta.normalization.cropSize ;
scale = max(fullImageSize ./ imageSize(1:2)) ;
imNorm = imresize(frame, scale, ...
              net.meta.normalization.interpolation, ...
              'antialiasing', false) ;
imNorm = bsxfun(@minus, imNorm, net.meta.normalization.averageImage) ;

boxes=single(detections')+1;
boxes = bsxfun(@times, boxes - 1, scale) + 1 ;
roi=[detections(:,3)' ; boxes];
% obtain the CNN otuput
net.conserveMemory = 0; 
net.eval({'data', imNorm, 'rois',roi}) ;
f = squeeze(gather(net.vars(net.getVarIndex('fc7x')).value)) ; 
f=f';

