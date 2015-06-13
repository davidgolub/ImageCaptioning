--[[

  Hidden Layer base class

--]]

local HiddenLayer = torch.class('imagelstm.HiddenLayer')

function HiddenLayer:__init(config)
   self.gpu_mode = config.gpu_mode or false
   self.image_dim = config.image_dim or 1024
   self.proj_dim = config.mem_dim or 300
   self.dropout = config.dropout and false or config.dropout
   self.num_layers = config.num_layers or 1 -- how many layers in the lstm
end

-- Returns all of the weights of this module
function HiddenLayer:getWeights()
end

-- Returns all the nn modules of this layer as an array
function HiddenLayer:getModules() 
end

-- Sets gpu mode
function HiddenLayer:set_gpu_mode()
end

function HiddenLayer:set_cpu_mode()
end

-- Enable Dropouts
function HiddenLayer:enable_dropouts()
end

-- Disable Dropouts
function HiddenLayer:disable_dropouts()
end

-- Does a single forward step of hidden layer, which
-- projects inputs into hidden state for lstm. Returns an array
-- Where first state corresponds to cell state, second state
-- corresponds to first hidden state
function HiddenLayer:forward(inputs)
end

-- Does a single backward step of hidden layer
-- Cell errors is an array where first input is error with respect to 
-- cell inputs of lstm, second input is error with respect to hidden inputs
-- of lstm
function HiddenLayer:backward(inputs, cell_errors)
end

-- Returns size of outputs of this hidden module
function HiddenLayer:getOutputSize()
end

-- Returns parameters of this model: parameters and gradients
function HiddenLayer:getParameters()
end

-- zeros out the gradients
function HiddenLayer:zeroGradParameters() 
end

function HiddenLayer:normalizeGrads(batch_size)
end

