require('.')

-- tests
include('tests/GradChecks.lua')
include('tests/GpuChecks.lua')

-- grad checks
grad_checker = imagelstm.GradChecks{}
grad_checker:check_lstm_captioner()

-- gpu checks
gpu_checker = imagelstm.GpuChecks{}
gpu_checker:check_gpu()