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
id_feat=[];
track=[];
non_init=true;
appear=false;
%struct for gt and  features of detections in 4 cameras 
for i=1:4
    data{i}=struct('gt',cam{i},'feature_set',[]);
end
% load image 
for k = 1 : num
    x=cell([],4);
    for i=1:4
        frame{i}= read(v{i},k); 
        x{i}=find(data{i}.gt(:,2)==k-1);% find whether there are detections in single frame
        if ~isempty(x{i})
            detections=data{i}.gt(x{i},:);
            % extract the CNN features (using VGG model)
            f=get_features(frame{i},detections,net);
            data{i}.feature_set(x{i},:)=f;            
            p=find(detections(:,3)==idf, 1);
            if ~isempty(p)
                if non_init  %initialize the cam and position of target
                    res=detections(find(detections(:,3)==idf, 1),:);
                    non_init=false;
                    appear=true;
                end
                id_feat=[id_feat;f(p,:)];
                id_feat_mean=sum(id_feat,1)/size(id_feat,1);
            end  
        end                          
    end
     % tracking
     if non_init
         go to disp_result
     else
         if appear
            [appear, res]=single_camera_tracker(data,res,idf,x);           
        else 
            [appear, res]=inter_camera_tracker(data,x,id_feat_mean);
        end
     end    
    track=[track;res];
    -------------------------------------------
    show detection
     disp_results    

end
    
    
end
