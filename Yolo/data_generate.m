function [weight , data] = data_generate(gtbox,gtlabel,imgsize)
% generate the training data in the Yolo format
%
location = [0:1:7]/7;
para.side = 7;
para.output = 30;
para.coord = 5.0;
para.noobj = 0.5;

temp1=zeros(para.output , para.side^2); % for weight
temp1(5,:) = para.noobj;
temp1(10,:) = para.noobj;
temp2=zeros(para.output , para.side^2); % for data

[row,clom]=size(gtbox);
yolobox = zeros(row,clom);
yolobox(:,1) = (gtbox(:,1) + gtbox(:,3))/2/imgsize(1);
yolobox(:,2) = (gtbox(:,2) + gtbox(:,4))/2/imgsize(2);
yolobox(:,3) = sqrt((gtbox(:,3) - gtbox(:,1))/imgsize(1));
yolobox(:,4) = sqrt((gtbox(:,4) - gtbox(:,2))/imgsize(1));

 for k=1:row
    grid_num1 = zeros(para.output,1);
    grid_num1(1:4)=ones(4,1)*para.coord;
    grid_num1(6:9)=ones(4,1)*para.coord;
    
    px = find(location<yolobox(k,1), 1, 'last' );
    py = find(location<yolobox(k,2), 1, 'last' );
    pc = gtlabel(k) + 10;
    tag = (py-1)*para.side+px; % index of grid    
    grid_num1(5:5:10) = 1;
    grid_num1(pc) = 1; % class probility    
    temp1(:,tag)=grid_num1; % weight
    
    grid_num1(1:4) = yolobox(k,:);
    grid_num1(6:9) = yolobox(k,:);
    temp2(:,tag)=grid_num1; % data
 end
weight = reshape( temp1 , para.output*para.side^2 , 1);
data = reshape( temp2 , para.output*para.side^2 , 1);



