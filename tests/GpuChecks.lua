--[[

  Image Captioning using an LSTM and a softmax for the language model

--]]

local GpuChecks = torch.class('imagelstm.GpuChecks')

function GpuChecks:__init(config)
  require('cutorch')
  require('cunn')
end

function GpuChecks:check_gpu()
  local inputs = torch.rand(1000)
  local net = nn.Linear(1000, 5000)

  local cpu_time = self:check_cpu_speed(inputs, net)
  local gpu_time = self:check_gpu_speed(inputs, net)

  print("Cpu time is ")
  print(cpu_time)

  print ("Gpu time is")
  print(gpu_time)
end

-- Checks how fast CPU speed is for neural net
function GpuChecks:check_cpu_speed(inputs, nnet)
  local start_time = sys.clock()
  for i = 1, 1000 do
    nnet:forward(inputs)
  end
  local end_time = sys.clock()
  return (end_time - start_time) / 1000
end

-- Checks how fast GPU speed is for neural net
function GpuChecks:check_gpu_speed(inputs, nnet)
  inputs = inputs:cuda()
  nnet:cuda()
  local start_time = sys.clock()
  for i = 1, 1000 do
    nnet:forward(inputs)
  end
  local end_time = sys.clock()
  return (end_time - start_time) / 1000
end



