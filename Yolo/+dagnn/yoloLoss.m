classdef yoloLoss < dagnn.Loss
% Yolo loss

  properties
    sigma = 1.
  end

  methods
    function outputs = forward(obj, inputs, params)
      delta = inputs{1} - inputs{3} ;
      % loss function
      outputs{1} = inputs{2}(:)' * delta(:).^2 ;

      % Accumulate loss statistics.
      n = obj.numAveraged ;
      m = n + gather(sum(inputs{2}(:))) + 1e-9 ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      delta = 2* (inputs{1} - inputs{3}) ;
%       delta = abs(delta) ;

      derInputs = {inputs{2} .* delta .* derOutputs{1}, [], []} ;
      derParams = {} ;
    end

    function obj = yoloLoss(varargin)
      obj.load(varargin) ;
      obj.loss = 'yoloLoss';
    end
  end
end
