function run_CLT()
%% Context learning based tracker across camera networks
%
%  Wei Liu (liuwei16@nudt.edu.cn); Jingjing Xiao (shine636363@sina.com)  Nov. 2016
% 

warning off
addpath(genpath('./'))
init_tracker

% train tracker at 1st frame
for idp = 0: 1: numP
    
    % read pedestrian detection
    ids      = find(det(:, 3) == idp);  
    active_c = det(ids(1), 1);        % active camera
    start_f  = det(ids(1), 2) + 1;    % starting frame
    
    % train tracker: extract CNN reference feature
    img      = read(v{active_c}, start_f); 
    bbox     = get_bbox(cam{active_c}(start_f, :));
    feat_ref = get_features(img, bbox, net);
 
    % tracking
    idf = start_f + 1;
    while idf < v{active_c}.NumberOfFrames
        
        % read detections
        if flag_single
            % read detections from one camera
            x = cam{active_c}(:,2) == idf;
            detections = get_bbox(cam{active_c}(x,:));            
        else
            % read detections from all cameras
            x = det(:,2) == idf;
            detections = get_bbox(det(x,:));
        end                 
        
        % extract CNN features from detections
        feat_obs = get_features(img, detections, net); 
        
        % matching Bhattacharyya distance
        scores = sum(sqrt(feat_obs.*repmat(feat_ref, size(feat_obs, 1), 1))');
        
        % find target id
        if flag_single            
            if max(scores(:)) < thresh_single
                % target disappears
                flag_single = 0;                 
                %----------------------------
                %  save as rest = [] or nan
                %----------------------------    
            else
                [~, idt] = max(scores);
                %-------------------------
                %  save as rest = XXXX
                %-------------------------                
            end
        else
            if max(scores(:)) > thresh_multi
                % target re-appears
                flag_single = 1;
                [~, idt]    = max(scores);
                active_c    = x(idt, 1);
                %-------------------------
                %  save as rest = XXXX
                %-------------------------         
            else
                %----------------------------
                %  save as rest = [] or nan
                %----------------------------       
            end            
        end    
        idf = idf + 1;
    end
    %------------------------------------------
    % save results of pedestrains in to *.mat 
    %------------------------------------------
end
