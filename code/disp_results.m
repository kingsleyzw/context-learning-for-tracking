%% disp the results within images
for i=1:4
    subplot(2,2,i);
    imshow(frame{i}),title(sprintf('Cam%d',i));
    text(2, 4, strcat('#',num2str(k-1)), 'Color','y', 'FontWeight','bold', 'FontSize',10);
    x=find(data{i}.gt(:,2)==k-1);
    if ~isempty(x)
        for j=1:size(x)
          rectangle('Position',data{i}.gt(x(j),([4,5,6,7])),'LineWidth',2,'EdgeColor','b');
          text(double(data{i}.gt(x(j),4)), double(data{i}.gt(x(j),5)) - 2, ...
                sprintf('%d',data{i}.gt(x(j),3)), ...
                'backgroundcolor', 'b', 'color', 'w', 'FontSize', 10);  
        end    
    end
    if ~isempty(res)&&res(1)==i&&res(3)==idf
       rectangle('Position',res([4,5,6,7]),'LineWidth',2,'EdgeColor','r');
          text(double(res(4)), double(res(5)) - 2, ...
                sprintf('%d',res(3)), ...
                'backgroundcolor', 'r', 'color', 'w', 'FontSize', 10);   
    end
end
pause(0.01)
clf  