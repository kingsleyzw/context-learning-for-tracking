function run_CLT()
%% Context learning based tracker across camera networks
%
%  Wei Liu (liuwei16@nudt.edu.cn); Jingjing Xiao (shine636363@sina.com)  Nov. 2016
% 

warning off
addpath(genpath('./'))
init_tracker
performance=[]; % for the whole performance of all targets
% train tracker at 1st frame
for idp = 10: 1: numP
    traj = [];% for the whole trajectory of the target
    resultPath=['./results/' num2str(idp) '_results.mat'];
    if exist(resultPath,'file')
       continue; 
    end
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
    flag_single=1;
    while idf < v{active_c}.NumberOfFrames-1
        feat_obs = [];        
        % read detections & extract cnn featues
        if flag_single
            % detections & features from one camera
            index    = find(cam{active_c}(:,2) == idf);           
            if ~isempty(index)
                bboxes   = get_bbox(cam{active_c}(index,:));
                image   = read(v{active_c}, idf + 1);                        
               feat_obs = get_features(image, bboxes{active_c}, net);
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
                        images{i} = read(v{i}, idf + 1); 
                        f         = get_features(images{i}, bboxes{i}, net); 
                        feat_obs  = [feat_obs ;f];
                    end               
                end
            else
                flag_single=0;
                scores = 0;
            end
            
        end                 
   
        % matching Bhattacharyya distance
%         scores = sum(sqrt(feat_obs.*repmat(feat_ref, size(feat_obs, 1), 1))');
%         scores = pdist2(feat_obs,feat_ref,'mahalanobis');
        if ~isempty(feat_obs)
            scores = sum(sqrt(feat_obs.*repmat(feat_ref, size(feat_obs, 1), 1))');
%             scores = 1-pdist2(feat_obs,feat_ref,'correlation');
        end        
        
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
        fprintf('%d:%d\n', idp, idf) ;
       
    end
    [precision, recall, F1_score] = per_eval(det, traj, idp); % performance evluation
    save(resultPath,'traj');  % save results 
    performance=[performance ; precision recall F1_score];    
end
save(['./results/performance'],'performance');

