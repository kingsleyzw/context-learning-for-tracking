function feat = get_features(images, detections, net)
%% extract the CNN features (using VGG model)
feat = [];
images = single(images);
%obtain the rois & CNN features
for i=1:size(detections,1)
    rois = imcrop(images,detections(i,:));
    rois = imresize(rois, net.meta.normalization.imageSize(1:2),'method',net.meta.normalization.interpolation) ;
    rois = bsxfun(@minus, rois, net.meta.normalization.averageImage) ;

    % obtain the CNN otuput
    net.conserveMemory = 0; 
    net.eval({'input',rois}) ;
    f_rois = squeeze(squeeze(gather(net.vars(net.getVarIndex('x17')).value))) ; 
    feat = [feat; f_rois'];
end
