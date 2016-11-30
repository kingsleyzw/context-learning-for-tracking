%% read detections & extract cnn featues

feat_obs = []; 
% read images from all cameras (for tracking result display )
for i = 1: 4      
   images{i} = read(v{i}, idf + 1);      
end

if flag_single
    % detections & features from one camera
    index    = find(cam{active_c}(:,2) == idf);           
    if ~isempty(index)
        bboxes   = get_bbox(cam{active_c}(index,:));       
        feat_obs = get_features(images{active_c}, bboxes{active_c}, net ,opts);
    else
        flag_single=0;
        scores = 0;
    end
else
    % detections & features from all cameras
    index = find(det(:,2) == idf);
    if ~isempty(index)
        bboxes = get_bbox(det(index,:));
        for i = 1: 4
            if ~isempty(bboxes{i})        
                f         = get_features(images{i}, bboxes{i}, net ,opts); 
                feat_obs  = [feat_obs ;f];
            end               
        end
    else
        flag_single=0;
        scores = 0;
    end            
end                 