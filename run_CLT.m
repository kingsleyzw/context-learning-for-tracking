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
    active_c = det(ids(1), 1);    % active camera
    start_f  = det(ids(1), 2);    % starting frame
    
    % train tracker: extract CNN reference feature
    img      = read(v{active_c}, start_f + 1); 
    bbox     = get_bbox(det(ids(1), :));
    feat_ref = get_features(img, bbox{active_c}, net);
 
    % tracking
    idf = start_f + 1;
    while idf < v{active_c}.NumberOfFrames-1
        feat_obs = [];
        
        % read detections & extract cnn featues
        if flag_single
            % detections & features from one camera
            images   = read(v{active_c}, idf + 1); 
            index    = find(cam{active_c}(:,2) == idf);
            bboxes   = get_bbox(cam{active_c}(index,:));            
            feat_obs = get_features(images, bboxes{active_c}, net); 
        else
            % detections & features from all cameras
            index = find(det(:,2) == idf);
            bboxes = get_bbox(det(index,:));
            for i = 1: 4
                if ~isempty(bboxes{i})        
                    images{i} = read(v{i}, idf + 1); 
                    f         = get_features(images{i}, bboxes{i}, net); 
                    feat_obs  = [feat_obs ;f];
                end               
            end
        end                 
   
        % matching Bhattacharyya distance
        scores = sum(sqrt(feat_obs.*repmat(feat_ref, size(feat_obs, 1), 1))');
        
        % find target id
        if flag_single            
            if max(scores(:)) < thresh_single
                % target disappears
                flag_single = 0;             
                rest        = []; 
            else
                [~, idt] = max(scores);
                rest     = cam{active_c}(index(idt),:);    
            end
        else
            if max(scores(:)) > thresh_multi
                % target re-appears
                flag_single = 1;
                [~, idt]    = max(scores);
                active_c    = det(index(idt),1);
                rest        = det(index(idt),:);
            else
                rest        = [];
            end            
        end    
        idf  = idf + 1;
        traj = [traj; rest];
    end 
    save(['./results/' num2str(idp) '_results'],'traj');  % save results
    [precision, recall] = per_eval(det, traj, idp);       % performance evluation
end
