function run_CLT()
%% Context learning based tracker across camera networks
%
%  Wei Liu (liuwei16@nudt.edu.cn); Jingjing Xiao (shine636363@sina.com)  Nov. 2016
% 

warning off
addpath(genpath('./'))
init_tracker
% load fast VGG
run(fullfile(fileparts(mfilename('fullpath')), ...
   'externel','matconvnet', 'matlab', 'vl_setupnn.m')) ;
opts.modelPath = fullfile(fileparts(mfilename('fullpath')), ...
'..', 'models', 'fast-rcnn-vgg16-pascal07-dagnn.mat') ;
opts.gpu = [] ;
net = load(opts.modelPath) ;
net = dagnn.DagNN.loadobj(net);
net.mode = 'test' ; 
% -------------------------
idf=0;
feature_set=cell([],4); % for features of detections in 4 cameras 
% load image 
for k = 1 : num
    x=cell([],4);
    for i=1:4
        frame{i}= read(v{i},k); 
        x{i}=find(cam{i}(:,2)==k-1);% find whether there are detections in single frame
        if ~isempty(x{i})
            detections=cam{i}(x{i},:);
            % extract the CNN features (using VGG model)
            f=get_features(frame{i},detections,net);
            feature_set{i}(x{i},:)=f;
            %initialize the cam and position of target
            if k==1&&~isempty(find(cam{i}(x{i},3)==idf, 1))
                init=cam{i}(1,:);
            end  
        end
                          
    end
    % tracking
    if k==1 % single camera racking
        res=single_camera_tracker( frame{init(1)} , init );
    else if ~isempty(find(cam{init(1)}(x{init(1)},3)==idf, 1))% target remains in original camera
             res=single_camera_tracker( frame{init(1)} , res );
        else % inter camera tracking
             res=inter_camera_tracker();
             init=res;
        end
    end
    
    %-------------------------------------------
    % show detection
     disp_results
end
