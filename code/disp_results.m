%% disp the results within images

subplot(2,2,1); 
imshow(frame{1}),title('Cam1');
text(2, 4, strcat('#',num2str(k-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10);
x=find(cam{1}(:,2)==k-1);
if ~isempty(x)
    for i=1:size(x)
      rectangle('Position',cam{1}(x(i),([4,5,6,7])),'LineWidth',2,'EdgeColor','b');
      text(double(cam{1}(x(i),4)), double(cam{1}(x(i),5)) - 2, ...
            sprintf('%d', cam{1}(x(i),3)), ...
            'backgroundcolor', 'b', 'color', 'w', 'FontSize', 10);  
    end    
end
subplot(2,2,2);
imshow(frame{2}),title('Cam2');
text(2, 4, strcat('#',num2str(k-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10);
x=find(cam{2}(:,2)==k-1);
if ~isempty(x)
    for i=1:size(x)
      rectangle('Position',cam{2}(x(i),([4,5,6,7])),'LineWidth',2,'EdgeColor','b');
      text(double(cam{2}(x(i),4)), double(cam{2}(x(i),5)) - 2, ...
            sprintf('%d', cam{2}(x(i),3)), ...
            'backgroundcolor', 'b', 'color', 'w', 'FontSize', 10);  
    end    
end
subplot(2,2,3);
imshow(frame{3}),title('Cam3');
text(2, 4, strcat('#',num2str(k-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10);
x=find(cam{3}(:,2)==k-1);
if ~isempty(x)
    for i=1:size(x)
      rectangle('Position',cam{3}(x(i),([4,5,6,7])),'LineWidth',2,'EdgeColor','b');
      text(double(cam{3}(x(i),4)), double(cam{3}(x(i),5)) - 2, ...
            sprintf('%d', cam{3}(x(i),3)), ...
            'backgroundcolor', 'b', 'color', 'w', 'FontSize', 10);  
    end    
end
subplot(2,2,4);
imshow(frame{4}),title('Cam4');
text(2, 4, strcat('#',num2str(k-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10);
x=find(cam{4}(:,2)==k-1);
if ~isempty(x)
    for i=1:size(x)
      rectangle('Position',cam{4}(x(i),([4,5,6,7])),'LineWidth',2,'EdgeColor','b');
      text(double(cam{4}(x(i),4)), double(cam{4}(x(i),5)) - 2, ...
            sprintf('%d', cam{4}(x(i),3)), ...
            'backgroundcolor', 'b', 'color', 'w', 'FontSize', 10);  
    end    
end
pause(0.01)
clf  