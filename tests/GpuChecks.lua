--[[

  Image Captioning using an LSTM and a softmax for the language model

--]]

local GpuChecks = torch.class('imagelstm.GpuChecks')

function GpuChecks:__init(config)
  require('cutorch')
  require('cunn')

  local in_dim = 400
  local mem_dim = 1500
  local num_classes = 4000
  local criterion = nn.ClassNLLCriterion()

  self.image_captioner = imagelstm.ImageCaptionerLSTM{
    in_dim  = in_dim,
    mem_dim = mem_dim,
    output_module_fn = self:new_caption_module(),
    criterion = criterion,
  }

  self.gpu_image_captioner = imagelstm.ImageCaptionerLSTM{
    gpu_mode = true,
    in_dim  = in_dim,
    mem_dim = mem_dim,
    output_module_fn = self:new_caption_module(),
    criterion = criterion,
  }

end

function GpuChecks:new_caption_module()
  local caption_module = nn.Sequential()

  caption_module
    :add(nn.Linear(1500, 4000))
    :add(nn.LogSoftMax())
  return caption_module
end

function GpuChecks:check_gpu()
  self:check_lstm_full_layer()
  self:check_lstm_captioner()
  self:check_nn_module()
end

function GpuChecks:check_lstm_full_layer()
  local in_dim = 300
  local mem_dim = 1500
  local input = torch.rand(10, 300)
  local num_iter = 20

  local lstm_gpu_layer = imagelstm.LSTM_Full{
    gpu_mode = true,
    in_dim  = in_dim,
    mem_dim = mem_dim,
  }

  local lstm_cpu_layer = imagelstm.LSTM_Full{
    gpu_mode = false,
    in_dim  = in_dim,
    mem_dim = mem_dim,
  }

  local gpu_time = self:check_gpu_speed(input, nil, lstm_gpu_layer, num_iter)
  local cpu_time = self:check_cpu_speed(input, nil, lstm_cpu_layer, num_iter)

  print("Cpu time for image captioner is")
  print(cpu_time)

  print ("Gpu time for image captioner is")
  print(gpu_time)
end

function GpuChecks:check_lstm_captioner()
  local input = torch.rand(10, 400)
  local output = torch.IntTensor(10)
  for i = 1, 10 do
    output[i] = i
  end

  local num_iter = 20
  local cpu_time = self:check_cpu_speed(input, output, self.image_captioner, num_iter)
  local gpu_time = self:check_gpu_speed(input, output, self.gpu_image_captioner, num_iter)

  print("Cpu time for image captioner is")
  print(cpu_time)

  print ("Gpu time for image captioner is")
  print(gpu_time)
end

function GpuChecks:check_nn_module()
  local inputs = torch.rand(1000)
  local net = nn.Linear(1000, 5000)
  local num_iter = 1000

  local cpu_time = self:check_cpu_speed(inputs, nil, net, num_iter)
  local gpu_time = self:check_gpu_speed(inputs, nil, net, num_iter)

  print("Cpu time for linear model is ")
  print(cpu_time)

  print ("Gpu time for linear model is ")
  print(gpu_time)
end

-- Checks how fast CPU speed is for neural net
function GpuChecks:check_cpu_speed(inputs, labels, nnet, num_iter)
  local start_time = sys.clock()
  for i = 1, num_iter do
      nnet:forward(inputs, labels)
  end
  local end_time = sys.clock()
  return (end_time - start_time) / num_iter
end

-- Checks how fast GPU speed is for neural net
function GpuChecks:check_gpu_speed(inputs, labels, nnet, num_iter)
  local inputs = inputs:cuda()
  nnet:forward(inputs, false)
  
  if labels ~= nil then
    labels = labels:cuda()
  end
  local start_time = sys.clock()
  for i = 1, num_iter do
      nnet:forward(inputs, labels)
  end
  local end_time = sys.clock()
  return (end_time - start_time) / num_iter
end



