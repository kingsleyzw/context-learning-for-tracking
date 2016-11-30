%% get tracking results

% matching Bhattacharyya distance
if ~isempty(feat_obs)
    scores = sum(sqrt(feat_obs.*repmat(feat_ref, size(feat_obs, 1), 1))');
%   scores = 1-pdist2(feat_obs,feat_ref,'correlation');
%   scores = pdist2(feat_obs,feat_ref,'mahalanobis');
end     
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