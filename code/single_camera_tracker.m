function [appear, res]=single_camera_tracker(data,res,idf,x)

p=find(data{res(1)}.gt(x{res(1)},3)==idf, 1);
if ~isempty(p)
    res=data{res(1)}.gt(x{res(1)}(p),:);
    appear=true;
else
    res=[];
%     res=[res([1,2,3]) 0 0 0 0];
    appear=false;
end  
