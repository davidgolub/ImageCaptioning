require('.')

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Testing parameters')
cmd:text('Options')
cmd:option('-in_dim', 150, 'input dimension into lstm')
cmd:option('-mem_dim', 300,'memory dimension into lstm')
cmd:option('-num_classes', 4000, 'number of classes')
cmd:option('-batch_size', 33, 'batch_size')
cmd:text()

-- parse input params
params = cmd:parse(arg)

-- tests
include('tests/GradChecks.lua')
include('tests/CpuChecks.lua')
include('tests/GpuChecks.lua')


-- cpu checks
cpu_checker = imagelstm.CpuChecks{}
cpu_checker:check_cpu()

-- gpu checks
gpu_checker = imagelstm.GpuChecks{}
gpu_checker:check_gpu()

-- grad checks
grad_checker = imagelstm.GradChecks{}
grad_checker:check_lstm_captioner()