function [appear, res]=inter_camera_tracker(data,x,id_feat_mean)

com_gt=[];
com_feat=[];

for i=1:4
    com_gt=[com_gt;data{i}.gt(x{i},:)];
    com_feat=[com_feat;data{i}.feature_set(x{i},:)];
end
dist=pdist2(com_feat,id_feat_mean,'cosine');
[score,index]=max(dist);
if score>0.9
    res=com_gt(index);
    appear=true;
else
    res=[];
%     res=[res([1,2,3]) 0 0 0 0];
    appear=false;
end

