function cboxes = bbox_std(boxes , imageSize)

x1 = boxes(1,:)*imageSize(2);
y1 = boxes(2,:)*imageSize(1);
x2 = x1 + boxes(3,:).^2*imageSize(2);
y2 = y1 + boxes(4,:).^2*imageSize(1);

cboxes = [x1 ; y1 ; x2 ; y2];