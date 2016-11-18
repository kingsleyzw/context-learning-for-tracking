classdef yoloLoss < dagnn.Loss


  properties
    sigma = 1.
  end

  methods
    function outputs = forward(obj, inputs, params)
      
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)

    end

    function obj = yoloLoss(varargin)
      obj.load(varargin) ;
      obj.loss = 'yoloLoss';
    end
  end
end
