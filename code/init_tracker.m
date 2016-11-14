%% initialise the tracker

% load videos
v{1}=VideoReader('../CLTdataset/dataset3-4cameras/Cam1.avi');
v{2}=VideoReader('../CLTdataset/dataset3-4cameras/Cam2.avi');
v{3}=VideoReader('../CLTdataset/dataset3-4cameras/Cam3.avi');
v{4}=VideoReader('../CLTdataset/dataset3-4cameras/Cam4.avi');
num = 0;
for i=1:4
    if  v{i}.NumberOfFrames>num
         num = v{i}.NumberOfFrames;
    end 
end

% load annotations
annotation_path='../CLTdataset/annotation_files/annotation/Dataset3/at least 4/';
annotation_files = dir([annotation_path '*.mat']);

%load cameras' detection
cam{1}=load('../CLTdataset/annotation_files/annotation/Dataset3/Cam1.dat');
cam{2}=load('../CLTdataset/annotation_files/annotation/Dataset3/Cam2.dat');
cam{3}=load('../CLTdataset/annotation_files/annotation/Dataset3/Cam3.dat');
cam{4}=load('../CLTdataset/annotation_files/annotation/Dataset3/Cam4.dat');
% for i=1:4
%     data{i}=struct('frameindex',[],'id',[],'gt',zeros(1,4),'feature_set',[]);
%     for j=1:size(cam{i},1)
%        data{i}(j).frameindex=cam{i}(j,2);
%        data{i}(j).id=cam{i}(j,3);
%     end
% end