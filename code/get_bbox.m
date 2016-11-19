function bboxes=get_bbox(detections)

bb=cell([],4);
for i=1:4
    idc= find(detections(:, 1) == i);
    if ~isempty(idc)
        bb{i}=[detections(idc,[4,5]) detections(idc,[6,7])];
%         bb{i}=[detections(idc,[4,5]) detections(idc,[4,5])+detections(idc,[6,7])];
    end
end

bboxes=bb;