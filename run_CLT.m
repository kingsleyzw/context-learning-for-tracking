function run_CLT()
%% Context learning based tracker across camera networks
%
%  Wei Liu (liuwei16@nudt.edu.cn); Jingjing Xiao (shine636363@sina.com)  Nov. 2016
% 

warning off
addpath(genpath('./'))
init_tracker
performance = []; % for the whole performance of all targets
% train tracker at 1st frame
for idp = 0: 1: numP
    traj = [];% for the whole trajectory of the target
    resultPath=['./results/' num2str(idp) '_results.mat'];
    if exist(resultPath,'file')
       continue; 
    end
    % read pedestrian detection
    ids      = find(det(:, 3) == idp);  
    active_c = det(ids(1), 1);    % active camera
    start_f  = det(ids(1), 2);    % starting frame
    
    %% train tracker: extract CNN reference feature
    img      = read(v{active_c}, start_f + 1); 
    bbox     = get_bbox(det(ids(1), :));
    feat_ref = get_features(img, bbox{active_c}, net ,opts);
 
    %% tracking
    idf = start_f + 1;
    flag_single = 1;
    while idf < v{active_c}.NumberOfFrames-1
               
        % read detections & extract cnn featues
        %------------------
        get_dets_features;
        %------------------ 
   
        % find target id
        %------------------
        get_tracking_rest;
        %------------------ 
        
        %display tracking results
        %------------------
         disp_results;
        %------------------   
        
        idf  = idf + 1;
        traj = [traj; rest];
        fprintf('%d:%d\n', idp, idf) ;
    end
    [precision, recall, F1_score] = per_eval(det, traj, idp); % performance evluation
    save(resultPath,'traj');  % save results 
    performance = [performance ; precision recall F1_score];    
end
save(['./results/performance'],'performance');

