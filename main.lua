require('.')


config.in_dim = 2
config.mem_dim = 10
config.num_classes = 4
imageLstm = imagelstm.LSTM_Full(config)

params, grad_params = imageLstm:parameters()

print("Params")
print(params)

print("Grad parameters")
print(grad_params[1])

inExample = torch.Tensor(5, config.in_dim)

out = imageLstm:forward(inExample, false)

caption_module = nn.Sequential()
caption_module
	:add(nn.Linear(config.mem_dim, config.num_classes))
    :add(nn.LogSoftMax())
res = caption_module:forward(out)

print(res)

-- negative log likelihood optimization objective
criterion = nn.ClassNLLCriterion()
desired = torch.Tensor{4, 3, 2, 1}

loss = criterion:forward(res, desired)

res_errs = criterion:backward(res, desired)

input_derivs = caption_module:backward(out, res)
in_errs = imageLstm:backward(inExample, input_derivs, false)

print("Grad parameters")
print(grad_params[1])

print("Derivatives")
print(in_errs)

