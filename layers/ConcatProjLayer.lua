--[[

  Concat layer: concatenates projected image features and word embeddings together
  after projecting image features to some dimension

--]]

local ConcatProjLayer = torch.class('imagelstm.ConcatProjLayer')

function ConcatProjLayer:__init(config)
   self.emb_learning_rate       = config.emb_learning_rate or 0.01
   self.gpu_mode = config.gpu_mode or false
   self.emb_dim = config.emb_dim or 300
   self.image_dim = config.image_dim or 1024
   self.proj_dim = config.proj_dim or 300
   self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)

   -- image feature embedding
   self.image_emb = nn.Linear(self.image_dim, self.proj_dim)
   self.combine_model = imagelstm.CRowJoinTable(2)

   self.emb.weight:copy(config.emb_vecs)

   local modules = nn.Parallel()
    :add(self.image_emb)
    :add(self.emb)

   self.params, self.grad_params = modules:getParameters()

   -- Copy the image embedding vectors
   if config.combine_weights ~= nil then
     self.params.weight:copy(config.combine_weights)
   end

  if gpu_mode then
    self:set_gpu_mode()
  end
end

-- Returns all of the weights of this module
function ConcatProjLayer:getWeights()
  return self.params
end

-- Sets gpu mode
function ConcatProjLayer:set_gpu_mode()
  self.image_emb:cuda()
  self.combine_model:cuda()
  self.emb:cuda()
end


-- Does a single forward step of concat layer, concatenating
-- Input 
function ConcatProjLayer:forward(word_indeces, image_feats)
   self.image_proj = self.image_emb:forward(image_feats)
   self.word_proj = self.emb:forward(word_indeces)
   res = self.combine_model:forward({self.word_proj, self.image_proj})
   return res
end

function ConcatProjLayer:backward(word_indices, image_feats, err)
   emb_errors = self.combine_model:backward({self.word_proj, self.image_proj}, err)

   -- get the image and word projection errors
   image_emb_errors = emb_errors[2]
   word_proj_errors = emb_errors[1]

   -- feed them backward
   self.image_emb:backward(image_feats, image_emb_errors)
   self.emb:backward(word_indices, word_proj_errors)
end

-- Returns size of outputs of this combine module
function ConcatProjLayer:getOutputSize()
  return self.emb_dim + self.proj_dim
end

function ConcatProjLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function ConcatProjLayer:zeroGradParameters() 
  self.emb:zeroGradParameters()
  self.image_emb:zeroGradParameters()
  self.combine_model:zeroGradParameters()
end

function ConcatProjLayer:normalizeGrads(batch_size)
  self.emb.gradWeight:div(batch_size)
  self.image_emb.gradWeight:div(batch_size)
end

function ConcatProjLayer:updateParameters()
    -- normalize gradients by batch size
  self.emb:updateParameters(self.emb_learning_rate)
  self.image_emb:updateParameters(self.emb_learning_rate)
end
