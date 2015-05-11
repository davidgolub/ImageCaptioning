require('.')

-- tests
include('tests/GradChecks.lua')

grad_checker = imagelstm.GradChecks{}

grad_checker:check_lstm_captioner()