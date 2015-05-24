require('.')

-- tests
include('tests/GradChecks.lua')
include('tests/CpuChecks.lua')
--include('tests/GpuChecks.lua')


-- cpu checks
cpu_checker = imagelstm.CpuChecks{}
cpu_checker:check_cpu()

-- gpu checks
gpu_checker = imagelstm.GpuChecks{}
gpu_checker:check_gpu()

-- grad checks
grad_checker = imagelstm.GradChecks{}
grad_checker:check_lstm_captioner()