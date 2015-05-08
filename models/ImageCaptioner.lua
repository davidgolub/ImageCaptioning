--[[

  Image Captioning using an LSTM and a softmax for the language model

--]]

local ImageCaptioner = torch.class('imagelstm.ImageCaptioner')

function ImageCaptioner:__init(config)
  self.reverse                 = config.reverse           or true
  self.image_dim               = config.image_dim         or 1024
  self.mem_dim                 = config.mem_dim           or 250
  self.learning_rate           = config.learning_rate     or 0.005
  self.emb_learning_rate       = config.emb_learning_rate or 0.1
  self.image_emb_learning_rate = config.emb_image_learning_rate or 0.1
  self.batch_size              = config.batch_size        or 100
  self.reg                     = config.reg               or 1e-4
  self.num_classes             = config.num_classes
  self.fine_grained            = (config.fine_grained == nil) and true or config.fine_grained
  self.dropout                 = (config.dropout == nil) and true or config.dropout

  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)

  -- image feature embedding
  self.image_emb = nn.Linear(self.image_dim, self.emb_dim)

  -- Do a linear combination of image and word features
  local x1 = nn.Identity()()
  local x2 = nn.Identity()()
  local a = imagelstm.CRowAddTable()({x1, x2})
  
  self.lstm_emb = nn.gModule({x1, x2}, {a})

  self.emb.weight:copy(config.emb_vecs)
  
  -- Copy the image embedding vectors
  if config.image_weights ~= nil then
    self.image_emb.weight:copy(config.image_weights)
  end

  self.in_zeros = torch.zeros(self.emb_dim)

  -- number of classes is equal to the vocab size sicne we are predicting new words
  self.num_classes = config.emb_vecs:size(1)

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- negative log likelihood optimization objective
  self.criterion = nn.ClassNLLCriterion()
  local num_caption_params = self:new_caption_module():getParameters():size(1)
  print("Number of caption parameters " .. num_caption_params)

  self.image_captioner = imagelstm.ImageCaptionerLSTM{
    in_dim  = self.emb_dim,
    mem_dim = self.mem_dim,
    output_module_fn = self:new_caption_module(),
    criterion = self.criterion,
  }

  self.params, self.grad_params = self.image_captioner:getParameters()
end

function ImageCaptioner:new_caption_module()
  local caption_module = nn.Sequential()
  if self.dropout then
    caption_module:add(nn.Dropout())
  end
  caption_module
    :add(nn.Linear(self.mem_dim, self.num_classes))
    :add(nn.LogSoftMax())

  return caption_module
end

function ImageCaptioner:train(dataset)
  self.image_captioner:training()
  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local feval = function(x)
      self.grad_params:zero()
      self.emb:zeroGradParameters()

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        -- get the image features
        local imgid = dataset.image_ids[idx]
        local image_feats = dataset.image_feats[imgid]

        -- get input and output sentences
        local sentence = dataset.sentences[idx]
        local out_sentence = dataset.pred_sentences[idx]

        -- get text/image inputs
        local text_inputs = self.emb:forward(sentence)
        local image_inputs = self.image_emb:forward(image_feats)

        -- get the lstm inputs
        local inputs = self.lstm_emb:forward({text_inputs, image_inputs})

        -- compute the loss
        local lstm_output, class_predictions, caption_loss = self.image_captioner:forward(inputs, out_sentence)
        loss = loss + caption_loss

        -- compute the input gradients with respect to the loss
        local input_grads = self.image_captioner:backward(inputs, lstm_output, class_predictions, out_sentence)

        -- backprop the gradients through the linear combination step
        local input_emb_grads = self.lstm_emb:backward({text_feats, image_feats}, input_grads)

        -- Separate gradients into word embedding and feature gradients
        local emb_grads = input_emb_grads[1]
        local image_grads = input_emb_grads[2]

        -- Do backward pass on image features and word embedding gradients
        self.emb:backward(sentence, emb_grads)
        self.image_emb:backward(image_feats, image_grads)
      end

      loss = loss / batch_size
      self.grad_params:div(batch_size)
      self.emb.gradWeight:div(batch_size)
      self.image_emb.gradWeight:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)

      print("Loss is ")
      print(loss)
      return loss, self.grad_params
    end

    optim.rmsprop(feval, self.params, self.optim_state)
    self.emb:updateParameters(self.emb_learning_rate)
    self.image_emb:updateParameters(self.image_emb_learning_rate)
  end
  xlua.progress(dataset.size, dataset.size)
end


function ImageCaptioner:predict(image_features, beam_size)
  -- TODO: Implement predicting image caption from image features with beam size
  -- set mode to predicting mode
  self.image_captioner:predicting()

  local tokens = {}

  -- Keep track of tokens predicted
  local num_iter = 0
  local tokens = {}

  -- Start with special START token:
  local next_token = torch.IntTensor{1}

  -- Terminate when predict the END token
  local end_token = torch.IntTensor{2}
  
  while next_token[1] ~= end_token[1] and num_iter < 20 do
     -- get text/image inputs

    local text_inputs = self.emb:forward(next_token)
    local image_inputs = self.image_emb:forward(image_features)

    -- get the lstm inputs
    local inputs = self.lstm_emb:forward({text_inputs, image_inputs})

    -- feed forward to predictions
    local class_predictions = torch.squeeze(self.image_captioner:forward(inputs))
    local pred_token = argmax(class_predictions)

    -- keep count of number of tokens seen already
    num_iter = num_iter + 1
    table.insert(tokens, pred_token)

    -- convert token into proper format for feed-forwarding
    next_token = torch.IntTensor{pred_token}
  end
  print(tokens)
  return tokens
end

function ImageCaptioner:predict_dataset(dataset)
  local predictions = {}
  for i = 1, 100 do
    xlua.progress(i, dataset.size)
    prediction = self:predict(dataset.image_feats[i])
    table.insert(predictions, prediction)
  end
  return predictions
end

function argmax(v)
  local idx = 1
  local max = v[1]
  for i = 2, v:size(1) do
    if v[i] > max then
      max = v[i]
      idx = i
    end
  end
  return idx
end

function ImageCaptioner:print_config()
  local num_params = self.params:size(1)
  local num_caption_params = self:new_caption_module():getParameters():size(1)
  printf('%-25s = %d\n', 'num params', num_params)
  printf('%-25s = %d\n', 'num compositional params', num_params - num_caption_params)
  printf('%-25s = %d\n', 'word vector dim', self.emb_dim)
  printf('%-25s = %d\n', 'image feature dim', self.image_dim)
  printf('%-25s = %d\n', 'Tree-LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n', 'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %.2e\n', 'word vector learning rate', self.emb_learning_rate)
  printf('%-25s = %.2e\n', 'image vector learning rate', self.image_emb_learning_rate)
  printf('%-25s = %s\n', 'dropout', tostring(self.dropout))
end

function ImageCaptioner:save(path)
  local config = {
    reverse           = self.reverse,
    batch_size        = self.batch_size,
    dropout           = self.dropout,
    emb_learning_rate = self.emb_learning_rate,
    emb_vecs          = self.emb.weight:float(),
    image_weights     = self.image_emb.weight:float(),
    fine_grained      = self.fine_grained,
    learning_rate     = self.learning_rate,
    mem_dim           = self.mem_dim,
    reg               = self.reg,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function ImageCaptioner.load(path)
  local state = torch.load(path)
  local model = imagelstm.ImageCaptioner.new(state.config)
  model.params:copy(state.params)
  return model
end
