function  truth = data_generate(gtbox,gtlabel,imgsize)

location = [0:1:7]/7;
para.side = 7; % there are 7*7 grid cells
para.output = 25; % 1+20+4

temp=zeros(para.output , para.side^2); % for truth

% convert the annotations to boxes in yolo style
[row,clom]=size(gtbox);
yolobox = zeros(row,clom);
yolobox(:,1) = (gtbox(:,1) + gtbox(:,3))/2/imgsize(1);
yolobox(:,2) = (gtbox(:,2) + gtbox(:,4))/2/imgsize(2);
yolobox(:,3) = (gtbox(:,3) - gtbox(:,1))/imgsize(1);
yolobox(:,4) = (gtbox(:,4) - gtbox(:,2))/imgsize(2);

 for k=1:row % row is the num of boxes
    grid_truth = zeros(para.output,1);   
    grid_truth(1) = 1; % confidence
    pc = gtlabel(k) + 1;
    grid_truth(pc) = 1; % class probability   
    
    px = find(location<yolobox(k,1), 1, 'last' );% px: index of col in 7*7
    py = find(location<yolobox(k,2), 1, 'last' );% py: index of row in 7*7
    grid_truth(22) = yolobox(k,1)*para.side-px+1;% x: offset of practical grid cell
    grid_truth(23) = yolobox(k,2)*para.side-py+1;% y: offset of practical grid cell
    grid_truth(24) = yolobox(k,3);               % w: relative to image size
    grid_truth(25) = yolobox(k,4);               % h: relative to image size

    tag = (py-1)*para.side+px; % index of grid cell     
    temp(:,tag) = grid_truth; % weight
 end
truth = reshape( temp , 1 , 1 , para.output*para.side^2 );




