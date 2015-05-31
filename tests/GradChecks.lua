--[[

  Image Captioning using an LSTM and a softmax for the language model

--]]

local GradChecks = torch.class('imagelstm.GradChecks')

function GradChecks:__init(config)
  self.tol = 1e-5
  self.in_dim = 4
  self.mem_dim = 3
  self.num_classes = 5
  self.reg = 0.5
  self.criterion = nn.ClassNLLCriterion()
  self.image_captioner = imagelstm.ImageCaptionerLSTM{
    in_dim  = self.in_dim,
    mem_dim = self.mem_dim,
    output_module_fn = self:new_caption_module(),
    criterion = self.criterion,
  }

  self.params, self.grad_params = self.image_captioner:getParameters()
end

function GradChecks:new_caption_module()
  local caption_module = nn.Sequential()

  caption_module
    :add(nn.Linear(self.mem_dim, self.num_classes))
    :add(nn.LogSoftMax())
  return caption_module
end

function GradChecks:check_addlayer()
  input = torch.rand(10, 4)
  output = torch.IntTensor{1, 2, 5, 4, 3, 4, 1, 2, 3, 5}
  
  local feval = function(x)
      self.grad_params:zero()

      -- compute the loss
      local lstm_output, class_predictions, caption_loss = self.image_captioner:forward(input, output)

      -- compute the input gradients with respect to the loss
      local input_grads = self.image_captioner:backward(input, lstm_output, class_predictions, output)

      -- regularization
      caption_loss = caption_loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)

      return caption_loss, self.grad_params
  end

  diff, DC, DC_est = optim.checkgrad(feval, self.params, 1e-6)
  print("Gradient error for lstm captioner is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end

function GradChecks:check_lstm_captioner()
  input = torch.rand(10, 4)
  output = torch.IntTensor{1, 2, 5, 4, 3, 4, 1, 2, 3, 5}
  
  local feval = function(x)
      self.grad_params:zero()

      -- compute the loss
      local lstm_output, class_predictions, caption_loss = self.image_captioner:forward(input, output)

      -- compute the input gradients with respect to the loss
      local input_grads = self.image_captioner:backward(input, lstm_output, class_predictions, output)

      -- regularization
      caption_loss = caption_loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)

      return caption_loss, self.grad_params
  end

  diff, DC, DC_est = optim.checkgrad(feval, self.params, 1e-6)
  print("Gradient error for lstm captioner is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end


function GradChecks:check_add_module()

  self.add_layer = imagelstm.AddLayer{
    emb_dim = 10,
    image_dim = 10
  }

  self.add_params, self.add_grads = self.add_layer:getParameters()
  local sentence_input = torch.IntTensor{1, 2, 5, 4, 3}
  local image_input = torch.rand(10)
  
  local desired = torch.rand(5, 10)
  self.mse_criterion = nn.MSECriterion()

  local feval = function(x)
      self.add_grads:zero()

      -- compute the loss
      local outputs = self.add_layer:forward(sentence_input, image_input)
      local err = self.mse_criterion:forward(outputs, desired)

      local d_outputs = self.mse_criterion:backward(outputs, desired)
      -- compute the input gradients with respect to the loss
      local input_grads = self.add_layer:backward(sentence_input, image_input, d_outputs)

      return err, self.add_grads
  end

  
  diff, DC, DC_est = optim.checkgrad(feval, self.add_params, 1e-6)
  print("Gradient error for add layer is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end


function GradChecks:check_concat_proj_module()
  self.add_layer = imagelstm.ConcatProjLayer{
    emb_dim = 10,
    image_dim = 10,
    proj_dim = 10
  }

  self.add_params, self.add_grads = self.add_layer:getParameters()
  local sentence_input = torch.IntTensor{1, 2, 5, 4, 3}
  local image_input = torch.rand(10)
  
  local desired = torch.rand(5, 20)
  self.mse_criterion = nn.MSECriterion()

  local feval = function(x)
      self.add_grads:zero()

      -- compute the loss
      local outputs = self.add_layer:forward(sentence_input, image_input)
      local err = self.mse_criterion:forward(outputs, desired)

      local d_outputs = self.mse_criterion:backward(outputs, desired)
      -- compute the input gradients with respect to the loss
      local input_grads = self.add_layer:backward(sentence_input, image_input, d_outputs)

      return err, self.add_grads
  end

  
  diff, DC, DC_est = optim.checkgrad(feval, self.add_params, 1e-6)
  print("Gradient error for concat projection module is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end

function GradChecks:check_add_module()

  self.add_layer = imagelstm.AddLayer{
    emb_dim = 10,
    image_dim = 10
  }

  self.add_params, self.add_grads = self.add_layer:getParameters()
  local sentence_input = torch.IntTensor{1, 2, 5, 4, 3}
  local image_input = torch.rand(10)
  
  local desired = torch.rand(5, 10)
  self.mse_criterion = nn.MSECriterion()

  local feval = function(x)
      self.add_grads:zero()

      -- compute the loss
      local outputs = self.add_layer:forward(sentence_input, image_input)
      local err = self.mse_criterion:forward(outputs, desired)

      local d_outputs = self.mse_criterion:backward(outputs, desired)
      -- compute the input gradients with respect to the loss
      local input_grads = self.add_layer:backward(sentence_input, image_input, d_outputs)

      return err, self.add_grads
  end

  
  diff, DC, DC_est = optim.checkgrad(feval, self.add_params, 1e-6)
  print("Gradient error for add layer is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end


function GradChecks:check_single_add_module()
  self.add_layer = imagelstm.SingleAddLayer{
    emb_dim = 10,
    image_dim = 10,
  }

  self.add_params, self.add_grads = self.add_layer:getParameters()
  local sentence_input = torch.IntTensor{1, 2, 5, 4, 3}
  local image_input = torch.rand(10)
  
  local desired = torch.rand(5, 10)
  self.mse_criterion = nn.MSECriterion()

  local feval = function(x)
      self.add_grads:zero()

      -- compute the loss
      local outputs = self.add_layer:forward(sentence_input, image_input)
      local err = self.mse_criterion:forward(outputs, desired)

      local d_outputs = self.mse_criterion:backward(outputs, desired)
      -- compute the input gradients with respect to the loss
      local input_grads = self.add_layer:backward(sentence_input, image_input, d_outputs)

      return err, self.add_grads
  end

  
  diff, DC, DC_est = optim.checkgrad(feval, self.add_params, 1e-6)
  print("Gradient error for single add module is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end

function GradChecks:check_concat_module()

  self.add_layer = imagelstm.ConcatLayer{
    emb_dim = 10,
    image_dim = 10
  }

  self.add_params, self.add_grads = self.add_layer:getParameters()
  local sentence_input = torch.IntTensor{1, 2, 5, 4, 3}
  local image_input = torch.rand(10)
  
  local desired = torch.rand(5, 20)
  self.mse_criterion = nn.MSECriterion()

  local feval = function(x)
      self.add_grads:zero()

      -- compute the loss
      local outputs = self.add_layer:forward(sentence_input, image_input)
      local err = self.mse_criterion:forward(outputs, desired)

      local d_outputs = self.mse_criterion:backward(outputs, desired)
      -- compute the input gradients with respect to the loss
      local input_grads = self.add_layer:backward(sentence_input, image_input, d_outputs)

      return err, self.add_grads
  end

  
  diff, DC, DC_est = optim.checkgrad(feval, self.add_params, 1e-6)
  print("Gradient error for concat module is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end

