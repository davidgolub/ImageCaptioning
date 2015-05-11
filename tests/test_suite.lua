require('..')

-- tests
include('GradChecks.lua')
grad_checker = imagelstm.GradChecks{}

-- check lstm captioner
grad_checker:check_lstm_captioner()

-- check loading data

-- check other stuff TODO