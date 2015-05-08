--[[

  An ImageLSTMCaptioner takes in two things as input: an LSTM cell and an output
  function for that cell that is the criterion.

--]]

local ImageLSTMCaptioner = torch.class('imagelstm.IMAGELSTMSentiment')

function ImageLSTMCaptioner:__init(config)
  -- parameters for lstm cell
  self.in_dim           =  config.in_dim 
  self.mem_dim          =  config.mem_dim           or 150
  self.output_module_fn =  config.output_module_fn
  self.criterion        =  config.criterion
  self.num_layers       =  config.num_layers        or 1
  self.lstm_layer =  imagelstm.LSTM_Full(config)

  local modules = nn.Parallel()
    :add(self.lstm_layer)
    :add(self.output_module_fn)
    
  self.params, self.grad_params = modules:getParameters()
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- reverse: if true, read the input from right to left (useful for bidirectional LSTMs).
-- labels: T x 1 tensor of desired indeces
-- Returns T x error tensor, all the intermediate hidden states of the LSTM
function ImageLSTMCaptioner:forward(inputs, labels): 
  local lstm_output = self.lstm_layer:forward(inputs, self.reverse)
  local class_predictions = self.output_module_fn:forward(lstm_output)
  local err = self.criterion:forward(class_predictions, labels)
  return lstm_output, class_predictions, err
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- reverse: True if reverse input, false otherwise
-- lstm_output: T x num_layers x num_hidden tensor
-- class_predictions: T x 1 tensor of predictions
-- labels: actual labels
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function ImageLSTMCaptioner:backward(inputs, lstm_output, class_predictions, labels):
  output_module_derivs = self.criterion:backward(class_predictions, labels)
  lstm_output_derivs = self.output_module_fn:backward(lstm_output, output_module_derivs)
  lstm_input_derivs = self.lstm_layer:backward(inputs, lstm_output_derivs, self.reverse)

  return lstm_input_derivs
end

function ImageCaptioner:predict(image_captioner)
  print("TODO")
end

function argmax(v)
  local idx = 1
  local max = v[1]
  for i = 2, v:size(1) do
    if v[i] > max then
      max = v[i]
      idx = i
    end
  end
  return idx
end

function ImageLSTMCaptioner:getParameters()
  return self.params, self.grad_params
end

function ImageLSTMCaptioner.load(path)
  local state = torch.load(path)
  local model = treelstm.TreeLSTMSentiment.new(state.config)
  model.params:copy(state.params)
  return model
end
