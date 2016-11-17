function [precision, recall]=per_eval(det,traj,idp)
% for precision 
id_det = det(det(:,3)==idp, :);
num_p=0;
for i = 2 : size(id_det,1)
    pos=ismember(traj,id_det(i,:),'rows');
    if ~isempty(find(pos==1, 1))
       num_p=num_p+1 ;
    end 
end
precision=num_p/size(id_det,1);
% for recall
num_r=size(traj(traj(:,3)==idp,:),1);
recall=num_r/size(traj,1);
