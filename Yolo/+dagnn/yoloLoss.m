classdef yoloLoss < dagnn.Loss


  properties
    n = 2;             %each grid cell predicts n bbx 
    classes = 20;      % num of classes
    coords = 4;        % num of coords
    side = 7;          % for 7*7 grid cells
    locations = 49;    % 49 grid cells
    noobj = single(0.5);       % weight of no object
    coord = single(5.0);       % weight of coordinates
    sqrt_wh = 1;       % indicate the use of sqrt of w and h  
    usegpu = 1;
  end

  methods
    function outputs = forward(obj, inputs, params)
      % Note: rows of prediction:  [side*side*classes][side*side*n][side*side*n*coords]
      %       rows of truth:       [1+classes+coords]*side*side  
      prediction = squeeze(inputs{1}); % prediction from last layer, example : 1470*64
      truth =      single(squeeze(inputs{2})); % truth as input, example : 1225*64 (1225 = 25*49) 
      batch =      size(prediction,2); % num of images for one computation
      delta = zeros(size(prediction)); % for the deviation between prediction and truth
      cost = 0;                        % for the whole cost of the loss function
      avg_iou = 0;                     % indicate the prediction of iou
      avg_cat = 0;                     % indicate the prediction of right category
      avg_allcat = 0;                  % indicate the prediction of all categories
      avg_obj = 0;                     % indicate the prediction of obj confidence
      avg_anyobj = 0;                  % indicate the prediction of obj/noobj confidence
      count = 0;                       % indicates how many objects in truth of the whole batch
      if obj.usegpu
          delta = gpuArray(delta);
          avg_iou = gpuArray(avg_iou);                     
          avg_cat = gpuArray(avg_cat);                     
          avg_allcat = gpuArray(avg_allcat);                
          avg_obj = gpuArray(avg_obj);                    
          avg_anyobj = gpuArray(avg_anyobj);                  
      end
      
      for b = 1 : batch
         for i= 1 : obj.locations
            truth_index = (i-1)*(1+obj.classes+obj.coords);
            is_obj = logical(truth(truth_index+1,b)) ;
            for j = 1 : obj.n
                p_index = obj.locations*obj.classes + (i-1)*obj.n +j;
                delta(p_index,b) = obj.noobj*(0-prediction(p_index,b));
                cost = cost+obj.noobj*prediction(p_index,b)^2;  % the 4th term in the loss function
                avg_anyobj = avg_anyobj+prediction(p_index,b);
            end
           
            if ~is_obj, continue; end
            class_index = (i-1)*obj.classes;
            for j = 1 : obj.classes
                delta(class_index+j,b) =  truth(truth_index+1+j,b)-prediction(class_index+j,b);
                cost = cost+delta(class_index+j,b)^2;  % the 5th term in the loss function    
                if logical(truth(truth_index+1+j,b)), avg_cat = avg_cat+prediction(class_index+j,b); end
                avg_allcat = avg_allcat+prediction(class_index+j,b);
            end            
            truth_box = get_box(truth(truth_index+1+obj.classes+1 : truth_index+1+obj.classes+4 , b));
            truth_box.x = truth_box.x/obj.side;
            truth_box.y = truth_box.y/obj.side;
            best_index = 0;
            best_iou = 0;
            best_rmse = 20;
            for j = 1 : obj.n
               box_index = obj.locations*(obj.classes + obj.n) + ((i-1)*obj.n+(j-1))*obj.coords; 
               pred_box = get_box(prediction(box_index+1 : box_index+4 , b));
               pred_box.x = pred_box.x/obj.side;
               pred_box.y = pred_box.y/obj.side;
               if obj.sqrt_wh, pred_box.w = pred_box.w^2; pred_box.h = pred_box.h^2; end 
               iou = get_box_iou(pred_box , truth_box);
               rmse = get_box_rmse(pred_box , truth_box);
               if best_iou > 0 || iou > 0
                    if iou > best_iou
                        best_iou = iou;
                        best_index = j;
                    end
               else
                    if(rmse < best_rmse)
                        best_rmse = rmse;
                        best_index = j;
                    end
               end               
            end
            bestbox_index =  obj.locations*(obj.classes + obj.n) + ((i-1)*obj.n+(best_index-1))*obj.coords; 
            tbox_index = truth_index+1+obj.classes;
            p_index = obj.locations*obj.classes + (i-1)*obj.n +best_index;
            delta(p_index,b) = double(1-prediction(p_index,b));
            cost = cost - obj.noobj*prediction(p_index,b)^2;
            cost = cost + delta(p_index,b)^2;    % the 3th term in the loss function   
            avg_obj = avg_obj + prediction(p_index,b);
            % the following four deltas are relative to the first two terms in the loss function
            delta(bestbox_index+1,b) = obj.coord*(truth(tbox_index+1,b) - prediction(bestbox_index+1,b));
            delta(bestbox_index+2,b) = obj.coord*(truth(tbox_index+2,b) - prediction(bestbox_index+2,b));
            delta(bestbox_index+3,b) = obj.coord*(truth(tbox_index+3,b) - prediction(bestbox_index+3,b));
            delta(bestbox_index+4,b) = obj.coord*(truth(tbox_index+4,b) - prediction(bestbox_index+4,b));
            if obj.sqrt_wh
                delta(bestbox_index+3,b) = obj.coord*(sqrt(truth(tbox_index+3,b)) - prediction(bestbox_index+3,b));
                delta(bestbox_index+4,b) = obj.coord*(sqrt(truth(tbox_index+4,b)) - prediction(bestbox_index+4,b));
            end
            cost = cost + (1-best_iou)^2;    % the 1-2 term in the loss function
            avg_iou = avg_iou +best_iou;
            count = count + 1; 
         end
      end
      outputs{1} = cost;
      delta = reshape (delta, 1,1,size(delta,1),size(delta,2));
      inputs{1} = delta;
      fprintf('Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n' , avg_iou/count, avg_cat/count, avg_allcat/(count*obj.classes), avg_obj/count, avg_anyobj/(batch*obj.locations*obj.n), count);
      % Accumulate loss statistics.
      k = obj.numAveraged ;
      m = k + batch + 1e-9 ;
%       m = k + (count+obj.locations*obj.n*batch/2) + 1e-9 ;
      obj.average = (k * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = { inputs{1} .* derOutputs{1}, []} ;
      derParams = {} ;
    end
    
    

    function obj = yoloLoss(varargin)
      obj.load(varargin) ;
      obj.loss = 'yoloLoss';
    end
  end
end
