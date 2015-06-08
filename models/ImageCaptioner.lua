--[[

  Image Captioning using an LSTM and a softmax for the language model

--]]

local ImageCaptioner = torch.class('imagelstm.ImageCaptioner')

function ImageCaptioner:__init(config)
  self.gpu_mode                = config.gpu_mode          or false
  self.reverse                 = config.reverse           or true
  self.image_dim               = config.image_dim         or 1024
  self.combine_module_type     = config.combine_module    or "embedlayer"
  self.hidden_module_type      = config.hidden_module     or "projlayer"
  self.mem_dim                 = config.mem_dim           or 150
  self.emb_dim                 = config.emb_dim           or 50
  self.learning_rate           = config.learning_rate     or 0.01
  self.batch_size              = config.batch_size        or 100
  self.reg                     = config.reg               or 1e-6
  self.emb_vecs                = config.emb_vecs          
  self.dropout                 = (config.dropout == nil) and false or config.dropout
  self.optim_method            = config.optim_method or optim.rmsprop
  self.num_classes             = config.num_classes or 2944
  self.optim_state             = config.optim_state or {learning_rate = self.learning_rate}
  self.num_layers              = config.num_layers or 1

  if config.emb_vecs ~= nil then
    self.num_classes = config.emb_vecs:size(1)
  end

  self.combine_layer = self:get_combine_layer(self.combine_module_type)
  
  self.hidden_layer = self:get_hidden_layer(self.hidden_module_type)
 
  self.in_zeros = torch.zeros(self.emb_dim)
  self.optim_state = { learningRate = self.learning_rate }
  
  -- negative log likelihood optimization objective
  local num_caption_params = self:new_caption_module():getParameters():size(1)
  print("Number of caption parameters " .. num_caption_params)

  self.image_captioner = imagelstm.ImageCaptionerLSTM{
    gpu_mode = self.gpu_mode,
    in_dim  = self.combine_layer:getOutputSize(),
    mem_dim = self.mem_dim,
    num_layers = self.num_layers,
    output_module_fn = self:new_caption_module(),
    criterion = nn.ClassNLLCriterion()
  }

  -- set gpu mode
  if self.gpu_mode then
    self:set_gpu_mode()
  end
  
  local captioner_modules = self.image_captioner:getModules()
  local combine_modules = self.combine_layer:getModules()
  local hidden_modules = self.hidden_layer:getModules()

  local modules = nn.Parallel()
  self:add_modules(modules, captioner_modules)
  self:add_modules(modules, combine_modules)
  self:add_modules(modules, hidden_modules)

  self.params, self.grad_params = modules:getParameters()
end

-- Returns input layer module corresponding to type specified by parameter
-- Combine_module_type: type of input layer to return
-- Requires: combine_module_type not nil, one of "addlayer", "concatlayer",
-- "singleaddlayer", "concatprojlayer", "embedlayer"
-- Returns: combine layer module corresponding to this layer
function ImageCaptioner:get_combine_layer(combine_module_type)
  assert(combine_module_type ~= nil)
  local layer
  if combine_module_type == "addlayer" then
    layer = imagelstm.AddLayer{
      gpu_mode = self.gpu_mode,
      emb_dim = self.emb_dim,
      image_dim = self.image_dim,
      vocab_size = self.num_classes,
      dropout = self.dropout
    }
  elseif combine_module_type == "concatlayer" then
    layer = imagelstm.ConcatLayer{
      gpu_mode = self.gpu_mode,
      emb_dim = self.emb_dim,
      image_dim = self.image_dim,
      vocab_size = self.num_classes,
      dropout = self.dropout
    }
  elseif combine_module_type == "singleaddlayer" then
    layer = imagelstm.SingleAddLayer{
      gpu_mode = self.gpu_mode,
      emb_dim = self.emb_dim,
      image_dim = self.image_dim,
      vocab_size = self.num_classes,
      dropout = self.dropout
    }
  elseif combine_module_type == "concatprojlayer" then
    layer = imagelstm.ConcatProjLayer{
      gpu_mode = self.gpu_mode,
      emb_dim = self.emb_dim,
      image_dim = self.image_dim,
      vocab_size = self.num_classes,
      dropout = self.dropout
    }
  elseif combine_module_type == "embedlayer" then
    layer = imagelstm.EmbedLayer{  
    emb_dim = self.emb_dim,
    num_classes = self.num_classes,
    gpu_mode = self.gpu_mode,
    dropout = self.dropout
    }
  else -- module not recognized
    error("Did not recognize input module type", combine_module_type)
  end
  return layer
end

-- Returns hidden layer module corresponding to type specified by parameter
-- Hidden_module_type: type of input layer to return
-- Requires: hidden_module_type not nil, one of "projlayer"
-- Returns: hidden layer module corresponding to this layer
function ImageCaptioner:get_hidden_layer(hidden_module_type)
  assert(hidden_module_type ~= nil)
  local layer
  if hidden_module_type == "projlayer" then
    layer = imagelstm.HiddenProjLayer{
      gpu_mode = self.gpu_mode,
      image_dim = self.image_dim,
      mem_dim = self.mem_dim,
      dropout = self.dropout
    }
  else 
    error("Did not recognize hidden module type", hidden_module_type)
  end
  return layer
end
-- adds modules into parallel network from module list
-- requires parallel_net is of type nn.parallel
-- requires module_list is an array of modules that is not null
-- modifies: parallel_net by adding modules into parallel net
function ImageCaptioner:add_modules(parallel_net, module_list)
  assert(parallel_net ~= nil)
  assert(module_list ~= nil)
  for i = 1, #module_list do
    curr_module = module_list[i]
    parallel_net:add(curr_module)
  end
end

-- Set all of the network parameters to gpu mode
function ImageCaptioner:set_gpu_mode()
  self.image_captioner:set_gpu_mode()
  self.combine_layer:set_gpu_mode()
  self.hidden_layer:set_gpu_mode()
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

-- enables dropouts on all layers
function ImageCaptioner:enable_dropouts()
  self.image_captioner:enable_dropouts()
  self.combine_layer:enable_dropouts()
  self.hidden_layer:enable_dropouts()
end

-- disables dropouts on all layers
function ImageCaptioner:disable_dropouts()
  self.image_captioner:disable_dropouts()
  self.combine_layer:disable_dropouts()
  self.hidden_layer:disable_dropouts()
end

function ImageCaptioner:train(dataset)
  self:enable_dropouts()

  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  local tot_loss = 0
  --dataset.size
  for i = 1, dataset.size, self.batch_size do --dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1
    
    currIndex = 0
    local feval = function(x)
      self.grad_params:zero()
      local start = sys.clock()
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
          sentence = sentence:cuda()
          out_sentence = out_sentence:cuda()
          image_feats = image_feats:cuda()
        end

        -- get text/image inputs
        local inputs = self.combine_layer:forward(sentence, image_feats)
        local hidden_inputs = self.hidden_layer:forward(image_feats)

        local lstm_output, class_predictions, caption_loss = 
        self.image_captioner:forward(inputs, hidden_inputs, out_sentence)
        
        loss = loss + caption_loss

        local input_grads, hidden_grads = 
        self.image_captioner:backward(inputs, hidden_inputs, lstm_output, class_predictions, out_sentence)
        
        -- do backward through input to lstm
        self.hidden_layer:backward(image_feats, hidden_grads)
        self.combine_layer:backward(sentence, image_feats, input_grads)
      end

      tot_loss = tot_loss + loss
      loss = loss / batch_size
      self.grad_params:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      --print(currIndex, " of ", self.params:size(1))
      --currIndex = currIndex + 1
      return loss, self.grad_params
    end
    -- check gradients for lstm layer
    --diff, DC, DC_est = optim.checkgrad(feval, self.params, 1e-7)
    --print("Gradient error for lstm captioner is")
    --print(diff)
    --assert(diff < 1e-5, "Gradient is greater than tolerance")

    self.optim_method(feval, self.params, self.optim_state)
  end
  average_loss = tot_loss / dataset.size
  xlua.progress(dataset.size, dataset.size)

  return average_loss
end

-- Evaluates model on dataset
-- Returns average loss
function ImageCaptioner:eval(dataset)
  self:disable_dropouts()

  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  local tot_loss = 0
  --dataset.size
  for i = 1, dataset.size, self.batch_size do --dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1
    
    currIndex = 0
    local feval = function(x)
      local start = sys.clock()
      local loss = 0
      for j = 1, batch_size do
        self.image_captioner:reset_depth()
        local idx = indices[i + j - 1]
        
        --local idx = i + j - 1
        -- get the image features
        local imgid = dataset.image_ids[idx]
        local image_feats = dataset.image_feats[imgid]

        -- get input and output sentences
        local sentence = dataset.sentences[idx]
        local out_sentence = dataset.pred_sentences[idx]


        if self.gpu_mode then
          sentence = sentence:cuda()
          out_sentence = out_sentence:cuda()
          image_feats = image_feats:cuda()
        end

        -- get text/image inputs
        local inputs = self.combine_layer:forward(sentence, image_feats)
        local hidden_inputs = self.hidden_layer:forward(image_feats)

        local lstm_output, class_predictions, caption_loss = 
        self.image_captioner:forward(inputs, hidden_inputs, out_sentence)
        
        loss = loss + caption_loss

        
      end

      tot_loss = tot_loss + loss
      loss = loss / batch_size
      self.grad_params:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      --print(currIndex, " of ", self.params:size(1))
      --currIndex = currIndex + 1
      return loss, self.grad_params
    end
    feval()
  end
  average_loss = tot_loss / dataset.size
  xlua.progress(dataset.size, dataset.size)
  return average_loss
end

function ImageCaptioner:predict(image_features, beam_size)
  if self.gpu_mode then
    image_features = image_features:cuda()
  end

  -- Keep track of tokens predicted
  local num_iter = 0
  local tokens = {}

  -- does one tick of lstm, input token and previous lstm state as input
  -- next_token: input into lstm
  -- curr_iter: current iteration
  -- prev_outputs: hidden state, cell state of lstm
  -- returns predicted token, its log likelihood, state of lstm, and all predictions

  local function lstm_tick(next_token, prev_outputs, curr_iter)
   local inputs = self.combine_layer:forward(next_token)

   -- feed forward to predictions
   local next_outputs, class_predictions = self.image_captioner:tick(inputs, prev_outputs)
   local squeezed_predictions = torch.squeeze(class_predictions)
   local predicted_token = argmax(squeezed_predictions, num_iter < 3)
   
   --print("Predicted token ")
   --print(predicted_token, next_token)
   local likelihood = squeezed_predictions[predicted_token]

   return predicted_token, likelihood, next_outputs, squeezed_predictions
  end

  -- Start with special START token:
  local next_token = torch.IntTensor{1}

  -- Terminate when predict the END token
  local end_token = torch.IntTensor{2}

  -- Initial hidden state/cell state values for lstm
  local prev_outputs = self.hidden_layer:forward(image_features)

  if beam_size < 2 then
    local ll = 0
    -- Greedy search
    while next_token[1] ~= end_token[1] and num_iter < 20 do
       -- get text/image inputs
      local pred_token, likelihood, next_outputs, class_predictions = 
                        lstm_tick(next_token, prev_outputs, num_iter)
    
      -- keep count of number of tokens seen already
      num_iter = num_iter + 1
      ll = ll + likelihood
      if pred_token ~= end_token then
        table.insert(tokens, pred_token)
      end

      -- convert token into proper format for feed-forwarding
      next_token = torch.IntTensor{pred_token}
      prev_outputs = next_outputs
    end
    return {{ll, tokens}}
  else
    -- beam search
    local beams = {}

    -- first get initial predictions
    local _, _, next_outputs, class_predictions = lstm_tick(next_token, prev_outputs, num_iter)
    
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

      local next_beams = {}

      for i = 1, #beams do
        local curr_beam = beams[i]

        -- get all predicted tokens so far
        local curr_tokens_list = curr_beam[2]
        local next_token = torch.IntTensor{curr_tokens_list[#curr_tokens_list]}

        -- If the next token is the end token, just add prediction to list
        if next_token[1] == 2 then
           -- hack: 
           table.insert(next_beams, curr_beam)
        else 
          local log_likelihood = curr_beam[1]
          local prev_outputs = curr_beam[3]

          -- first get initial predictions
          local pred_token, likelihood, next_outputs, class_predictions = 
          lstm_tick(next_token, prev_outputs, num_iter)

          -- hack: for some reason tokens repeat on first iteration
          if num_iter > 0 then 
            table.insert(curr_tokens_list, pred_token)
          end

          local next_ll = log_likelihood + likelihood

          local next_beam = {next_ll, curr_tokens_list, next_outputs}
          table.insert(next_beams, next_beam)
        end
      end

      num_iter = num_iter + 1
      -- Keep top beam_size entries in beams
      beams = topk(next_beams, beam_size)
    end
    return beams
  end
end


function ImageCaptioner:predict_dataset(dataset, beam_size, num_predictions)
  self.image_captioner:disable_dropouts()
  local beam_size = beam_size or 1
  local predictions = {}
  num_predictions = num_predictions or 30
  for i = 1, num_predictions do
    xlua.progress(i, dataset.size)
    prediction = self:predict(dataset.image_feats[i], beam_size)
    table.insert(predictions, prediction)
  end
  return predictions
end

-- saves prediction to specified file path
function ImageCaptioner:save_predictions(predictions_save_path, loss, test_predictions)
  local predictions_file, err = io.open(predictions_save_path,"w")

  print('writing predictions to ' .. predictions_save_path)
  --predictions_file:write("LOSS " .. loss .. '\n')
  for i = 1, #test_predictions do
    local test_prediction = test_predictions[i]
    local test_prediction = test_predictions[i][1]
    local likelihood = test_prediction[1]
    local tokens = test_prediction[2]
    local sentence = table.concat(vocab:tokens(tokens), ' ')
    predictions_file:write(sentence .. '\n')
  end
  predictions_file:close()
end

function ImageCaptioner:print_config()
  local num_params = self.params:size(1)
  local num_caption_params = self:new_caption_module():getParameters():size(1)
  printf('%-25s = %d\n', 'num params', num_params)
  printf('%-25s = %d\n', 'num compositional params', num_params - num_caption_params)
  printf('%-25s = %d\n', 'word vector dim', self.emb_dim)
  printf('%-25s = %d\n', 'LSTM memory dim', self.mem_dim)
  printf('%-25s = %d\n', 'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n', 'number of classes', self.num_classes)
  printf('%-25s = %d\n', 'number of layers in lstm', self.num_layers)
  printf('%-25s = %s\n', 'combine module type', self.combine_module_type)
  printf('%-25s = %s\n', 'hidden module type', self.hidden_module_type)
  printf('%-25s = %s\n', 'dropout', tostring(self.dropout))
end

function ImageCaptioner:save(path)
  local config = {
    reverse           = self.reverse,
    batch_size        = self.batch_size,
    dropout           = self.dropout,
    num_classes       = self.num_classes,
    learning_rate     = self.learning_rate,
    mem_dim           = self.mem_dim,
    reg               = self.reg,
    emb_dim           = self.emb_dim,
    emb_vecs          = self.emb_vecs,
    optim_method      = self.optim_method,
    combine_module    = self.combine_module_type,
    hidden_module     = self.hidden_module_type,
    num_layers        = self.num_layers
  }

  torch.save(path, {
    params = self.params,
    optim_state = self.optim_state,
    config = config,
  })
end

-- returns model path representation based on the model configuration
-- epoch: model epoch to return
function ImageCaptioner:getPath(epoch) 
  local model_save_path = string.format(
  '/image_captioning_lstm.hidden_type_%s.input_type_%s.emb_dim_%d.num_layers_%d.mem_dim_%d.epoch_%d.th', 
  self.hidden_module_type,
  self.combine_module_type,
  self.emb_dim, self.num_layers,
  self.mem_dim, epoch)
  return model_save_path
end

function ImageCaptioner.load(path)
  local state = torch.load(path)
  local model = imagelstm.ImageCaptioner.new(state.config)
  print(state.params)
  print(state.optim_state)
  model.params:copy(state.params)
  model.optim_state = state.optim_state
  return model
end
