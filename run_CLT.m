function run_CLT()
%% Context learning based tracker across camera networks
%
%  Wei Liu (liuwei16@nudt.edu.cn); Jingjing Xiao (shine636363@sina.com)  Nov. 2016
% 

warning off
addpath(genpath('./'))
init_tracker

% train tracker at 1st frame
for idP = 0: 1: numP
    
    % read init pedestrian detection
    ids      = find(det(:, 3) == idP);  
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
            x=  cam{active_c}(:,2)==idf-1;
            detections =get_bbox(cam{active_c}(x,:));
            
        else
            % read detections from all cameras
            x=  det(:,2)==idf-1;
            detections =get_bbox(det(x,:));
        end                 
        % extract CNN features
        
        % matching
        
        % find id
        
        
    end
end

% % load image 
% for k = 1 : num
%     x = cell([],4); 
%     for i=1:4
%         frame{i}= read(v{i},k); 
%         x{i}=find(data{i}.gt(:,2)==k-1);% find whether there are detections in single frame
%         if ~isempty(x{i})
%             detections=data{i}.gt(x{i},:);
%             % extract the CNN features (using VGG model)
%             f = get_features(frame{i},detections, net);
%             data{i}.feature_set(x{i},:)=f;            
%             p=find(detections(:,3)==idf, 1);
%             if ~isempty(p)
%                 if non_init  %initialize the cam and position of target
%                     res=detections(find(detections(:,3)==idf, 1),:);
%                     non_init=false;
%                     appear=true;
%                 end
%                 id_feat=[id_feat;f(p,:)];
%                 id_feat_mean=sum(id_feat,1)/size(id_feat,1);
%             end  
%         end                          
%     end
%      % tracking
%      if non_init
%          go to disp_result
%      else
%          if appear
%             [appear, res]=single_camera_tracker(data,res,idf,x);           
%         else 
%             [appear, res]=inter_camera_tracker(data,x,id_feat_mean);
%         end
%      end    
% %      track=[track;res];
%     disp_results    
% end  
%     
% end
