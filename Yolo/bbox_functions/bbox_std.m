function cboxes = bbox_std(boxes , imageSize)

ctr_x = boxes(1,:)*imageSize(2);
ctr_y = boxes(2,:)*imageSize(1);
w = boxes(3,:).^2*imageSize(2);
h = boxes(4,:).^2*imageSize(1);
x1 = ctr_x - w/2;
y1 = ctr_y - h/2;
x2 = ctr_x + w/2;
y2 = ctr_y + h/2;
% x1 = boxes(1,:)*imageSize(2);
% y1 = boxes(2,:)*imageSize(1);
% x2 = x1 + boxes(3,:).^2*imageSize(2);
% y2 = y1 + boxes(4,:).^2*imageSize(1);

cboxes = [x1 ; y1 ; x2 ; y2];