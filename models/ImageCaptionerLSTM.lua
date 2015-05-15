--[[

  An ImageCaptionerLSTM takes in two things as input: an LSTM cell and an output
  function for that cell that is the criterion.

--]]

local ImageCaptionerLSTM = torch.class('imagelstm.ImageCaptionerLSTM')

function ImageCaptionerLSTM:__init(config)
  -- parameters for lstm cell
  self.gpu_mode = config.gpu_mode
  self.criterion        =  config.criterion
  self.output_module_fn = config.output_module_fn
  self.lstm_layer =  imagelstm.LSTM_Full(config)
  self.train_mode = true

  local modules = nn.Parallel()
    :add(self.lstm_layer)
    :add(self.output_module_fn)
    
  if self.gpu_mode then
    modules:cuda()
    self.criterion:cuda()
  end
  
  self.params, self.grad_params = modules:getParameters()
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- states: hidden, cell states of LSTM if true, read the input from right to left (useful for bidirectional LSTMs).
-- labels: T x 1 tensor of desired indeces
-- Returns lstm output, class predictions, and error if train, else not error 
function ImageCaptionerLSTM:forward(inputs, labels)
    local lstm_output = self.lstm_layer:forward(inputs, self.reverse)
    local class_predictions = self.output_module_fn:forward(lstm_output)
    local err = self.criterion:forward(class_predictions, labels)
    return lstm_output, class_predictions, err
end

-- Single tick of LSTM Captioner
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- states: hidden, cell states of LSTM if true, read the input from right to left (useful for bidirectional LSTMs).
-- labels: T x 1 tensor of desired indeces
-- Returns lstm output, class predictions, and error if train, else not error 
function ImageCaptionerLSTM:tick(inputs, states)
    local lstm_output = self.lstm_layer:tick(inputs, states)
    local ctable, htable = unpack(lstm_output)
    local hidden_state = htable
    local class_predictions = self.output_module_fn:forward(hidden_state)
    return lstm_output, class_predictions
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- reverse: True if reverse input, false otherwise
-- lstm_output: T x num_layers x num_hidden tensor
-- class_predictions: T x 1 tensor of predictions
-- labels: actual labels
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function ImageCaptionerLSTM:backward(inputs, lstm_output, class_predictions, labels)
  output_module_derivs = self.criterion:backward(class_predictions, labels)
  lstm_output_derivs = self.output_module_fn:backward(lstm_output, output_module_derivs)
  lstm_input_derivs = self.lstm_layer:backward(inputs, lstm_output_derivs, self.reverse)

  return lstm_input_derivs
end

-- Sets all networks to gpu mode
function ImageCaptionerLSTM:set_gpu_mode()
  self.criterion:cuda()
  self.output_module_fn:cuda()
  self.lstm_layer:cuda()
end

-- Sets all networks to cpu mode
function ImageCaptionerLSTM:set_cpu_mode()
  print("TODO")
end

function ImageCaptionerLSTM:getParameters()
  return self.params, self.grad_params
end

