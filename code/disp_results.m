%% disp the results within images
for i=1:4
    subplot(2,2,i);
    imshow(images{i}),title(sprintf('Cam%d',i));
    text(2, 4, strcat('#',num2str(idf-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10);
    x=find(cam{i}(:,2)==idf-1);
    if ~isempty(x)
        for j=1:size(x)
          rectangle('Position',cam{i}(x(j),([4,5,6,7])),'LineWidth',2,'EdgeColor','b');
          text(double(cam{i}(x(j),4)), double(cam{i}(x(j),5)) - 2, ...
                sprintf('%d',cam{i}(x(j),3)), ...
                'backgroundcolor', 'b', 'color', 'w', 'FontSize', 10);  
        end    
    end
    if ~isempty(rest)&&rest(1)==i
       rectangle('Position',rest([4,5,6,7]),'LineWidth',2,'EdgeColor','r');
          text(double(rest(4)), double(rest(5)) - 2, ...
                sprintf('%d',rest(3)), ...
                'backgroundcolor', 'r', 'color', 'w', 'FontSize', 10);   
    end
end
pause(0.01)
clf  