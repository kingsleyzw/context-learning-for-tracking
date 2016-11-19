function [precision, recall, F1_score] = per_eval(det, traj, idp)
% for precision 
num_p=size(traj(traj(:,3)==idp,:),1);
precision=num_p/size(traj,1);

% for recall
id_det = det(det(:,3)==idp, :);
num_r=0;
for i = 2 : size(id_det,1)
    pos=ismember(traj,id_det(i,:),'rows');
    if ~isempty(find(pos==1, 1))
       num_r=num_r+1 ;
    end 
end
recall=num_r/size(id_det,1);

% for F1_score
F1_score=2*precision*recall/(precision+recall);
