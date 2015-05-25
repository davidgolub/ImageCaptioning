--[[

  Image Captioning using an LSTM and a softmax for the language model

--]]

local ImageCaptioner = torch.class('imagelstm.ImageCaptioner')

function ImageCaptioner:__init(config)
  self.gpu_mode                = config.gpu_mode          or false
  self.reverse                 = config.reverse           or true
  self.image_dim               = config.image_dim         or 1024
  self.mem_dim                 = config.mem_dim           or 150
  self.learning_rate           = config.learning_rate     or 0.01
  self.emb_learning_rate       = config.emb_learning_rate or 0.01
  self.image_emb_learning_rate = config.emb_image_learning_rate or 0.01
  self.batch_size              = config.batch_size        or 100
  self.reg                     = config.reg               or 1e-4
  self.num_classes             = config.num_classes
  self.dropout                 = (config.dropout == nil) and false or config.dropout

  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)

  -- image feature embedding
  self.image_emb = nn.Linear(self.image_dim, self.emb_dim)

  -- Do a linear combination of image and word features
  local x1 = nn.Identity(self.emb_dim)()
  local x2 = nn.Identity(self.emb_dim)()
  local a = imagelstm.CRowAddTable()({x1, x2})
  
  self.lstm_emb = nn.gModule({x1, x2}, {a})

  self.emb.weight:copy(config.emb_vecs)
  
  -- Copy the image embedding vectors
  if config.image_weights ~= nil then
    self.image_emb.weight:copy(config.image_weights)
  end

  self.optim_state = { learningRate = self.learning_rate }

  self.in_zeros = torch.zeros(self.emb_dim)

  -- number of classes is equal to the vocab size size we are predicting new words
  self.num_classes = config.emb_vecs:size(1)

  -- negative log likelihood optimization objective
  local num_caption_params = self:new_caption_module():getParameters():size(1)
  print("Number of caption parameters " .. num_caption_params)

  self.image_captioner = imagelstm.ImageCaptionerLSTM{
    gpu_mode = self.gpu_mode,
    in_dim  = self.emb_dim,
    mem_dim = self.mem_dim,
    output_module_fn = self:new_caption_module(),
    criterion = nn.ClassNLLCriterion()
  }

  -- set gpu mode
  if self.gpu_mode then
    self:set_gpu_mode()
  end
  
  self.params, self.grad_params = self.image_captioner:getParameters()
  

end

-- Set all of the network parameters to gpu mode
function ImageCaptioner:set_gpu_mode()
  self.image_captioner:set_gpu_mode()
  self.lstm_emb:cuda()
  self.emb:cuda()
  self.image_emb:cuda()
end

function ImageCaptioner:set_cpu_mode()
  print("TODO")
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
  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  local tot_loss = 0
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local feval = function(x)
      self.grad_params:zero()
      self.emb:zeroGradParameters()
      self.image_emb:zeroGradParameters()

      local start = sys.clock()
      local tot_forward_diff = 0
      local tot_backward_diff = 0
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        
        --local idx = i + j - 1
        -- get the image features
        local imgid = dataset.image_ids[idx]
        local image_feats = dataset.image_feats[imgid]

        -- get input and output sentences
        local sentence = dataset.sentences[idx]
        local out_sentence = dataset.pred_sentences[idx]


        if self.gpu_mode then
          --sentence = sentence:cuda()
          --out_sentence = out_sentence
          image_feats = image_feats:cuda()
        end

        local start1 = sys.clock()
        -- get text/image inputs
        local text_inputs = self.emb:forward(sentence)

        local start2 = sys.clock()
        local image_inputs = self.image_emb:forward(image_feats)

        -- get the lstm inputs
        local start3 = sys.clock()
        local inputs = self.lstm_emb:forward({text_inputs, image_inputs})

        print(inputs)
        print(out_sentence)
        -- compute the loss
        local start4 = sys.clock()
        local lstm_output, class_predictions, caption_loss = self.image_captioner:forward(inputs, out_sentence)
        loss = loss + caption_loss

        -- compute the input gradients with respect to the loss
        local start5 = sys.clock()
        local input_grads = self.image_captioner:backward(inputs, lstm_output, class_predictions, out_sentence)

        -- backprop the gradients through the linear combination step
        local start6 = sys.clock()

        tot_forward_diff = tot_forward_diff + start5 - start4
        tot_backward_diff = tot_backward_diff + start6 - start5
        local input_emb_grads = self.lstm_emb:backward({text_feats, image_feats}, input_grads)

        -- Separate gradients into word embedding and feature gradients
        local emb_grads = input_emb_grads[1]
        local image_grads = input_emb_grads[2]

        -- Do backward pass on image features and word embedding gradients
        local start7 = sys.clock()
        self.emb:backward(sentence, emb_grads)
        self.image_emb:backward(image_feats, image_grads)
    
        --  start5 - start4, start4 - start3, start3 - start2, start2 - start1)
      end

      local start8 = sys.clock()
      print("Times are ", (start8 - start))
      print("Forward time ", tot_forward_diff)
      print("Backward time ", tot_backward_diff)
      tot_loss = tot_loss + loss
      loss = loss / batch_size
      self.grad_params:div(batch_size)
      self.emb.gradWeight:div(batch_size)
      self.image_emb.gradWeight:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)

      print("Caption loss is:")
      print(loss)

      return loss, self.grad_params
    end

    optim.rmsprop(feval, self.params, self.optim_state)
    self.emb:updateParameters(self.emb_learning_rate)
    self.image_emb:updateParameters(self.image_emb_learning_rate)
  end
  average_loss = tot_loss / dataset.size
  xlua.progress(dataset.size, dataset.size)

  return average_loss
end

function ImageCaptioner:predict(image_features, beam_size)
  if self.gpu_mode then
    image_features = image_features:cuda()
  end
  
  local image_inputs = self.image_emb:forward(image_features)

  -- Keep track of tokens predicted
  local num_iter = 0
  local tokens = {}

  -- does one tick of lstm, input token and previous lstm state as input
  -- next_token: input into lstm
  -- prev_outputs: hidden state, cell state of lstm
  -- returns predicted token, its log likelihood, state of lstm, and all predictions

  local function lstm_tick(next_token, prev_outputs)
   local text_inputs = self.emb:forward(next_token)
   -- get the lstm inputs
   local inputs = self.lstm_emb:forward({text_inputs, image_inputs})

   -- feed forward to predictions
   local next_outputs, class_predictions = self.image_captioner:tick(inputs, prev_outputs)

   local class_predictions = torch.squeeze(class_predictions)
   local pred_token = argmax(class_predictions, num_iter < 3)
   local likelihood = class_predictions[pred_token]

   return pred_token, likelihood, next_outputs, class_predictions
  end

  -- Start with special START token:
  local next_token = torch.IntTensor{1}

  -- Terminate when predict the END token
  local end_token = torch.IntTensor{2}

  -- Initial hidden state/cell state values for lstm
  local prev_outputs = nil

  if beam_size == 0 then
    local ll = 0
    -- Greedy search
    while next_token[1] ~= end_token[1] and num_iter < 20 do
       -- get text/image inputs
      local pred_token, likelihood, next_outputs, class_predictions = lstm_tick(next_token, prev_outputs)
    
      -- keep count of number of tokens seen already
      num_iter = num_iter + 1
      ll = ll + likelihood
      if pred_token ~= end_token then
        table.insert(tokens, pred_token)
      end

      print("On iteration %d", num_iter)
      print(pred_token)
      -- convert token into proper format for feed-forwarding
      next_token = torch.IntTensor{pred_token}
      prev_outputs = next_outputs
    end
    return {ll, tokens}
  else
    -- beam search
    local beams = {}

    -- first get initial predictions
    local _, _, next_outputs, class_predictions = lstm_tick(next_token, prev_outputs)
    
    -- then get best top k indices indices
    local best_indices = topkargmax(class_predictions, beam_size)

    -- then initialize our tokens list
    for i = 1, beam_size do
      local next_token = best_indices[i]
      local curr_beam = {class_predictions[next_token], {next_token}, next_outputs}
      table.insert(beams, curr_beam)
    end

    -- now do the general beam search algorithm
    while num_iter < 20 do
      num_iter = num_iter + 1

      local next_beams = {}

      for i = 1, #beams do
        local curr_beam = beams[i]

        -- get all predicted tokens so far
        local curr_tokens_list = curr_beam[2]
        local next_token = torch.IntTensor{curr_tokens_list[#curr_tokens_list]}

        -- If the next token is the end token, just add prediction to list
        if next_token[1] == 2 then
           table.insert(next_beams, curr_beam)
        else 
          local log_likelihood = curr_beam[1]
          local prev_outputs = curr_beam[3]

          -- first get initial predictions
          local pred_token, likelihood, next_outputs, class_predictions = lstm_tick(next_token, prev_outputs)
          table.insert(curr_tokens_list, pred_token)
          local next_ll = log_likelihood + likelihood

          local next_beam = {next_ll, curr_tokens_list, next_outputs}
          table.insert(next_beams, next_beam)
        end
      end

      -- Keep top beam_size entries in beams
      beams = topk(next_beams, beam_size)
    end
    return beams
  end
end


function ImageCaptioner:predict_dataset(dataset)
  local predictions = {}
  num_predictions = 1000 -- = dataset.size
  for i = 1, num_predictions do
    xlua.progress(i, dataset.size)
    prediction = self:predict(dataset.image_feats[i], 1)
    table.insert(predictions, prediction)
  end
  return predictions
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
    optim_state = self.optim_state,
    config = config,
  })
end

function ImageCaptioner.load(path)
  local state = torch.load(path)
  local model = imagelstm.ImageCaptioner.new(state.config)
  model.params:copy(state.params)
  model.optim_state = state.optim_state
  return model
end
