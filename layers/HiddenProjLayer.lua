--[[

  Hidden Project Layer: Projects image input into projection dimension twice. For feeding in
  image input into lstm
--]]

local HiddenProjLayer = torch.class('imagelstm.HiddenProjLayer')

function HiddenProjLayer:__init(config)
   self.emb_learning_rate       = config.emb_learning_rate or 0.01
   self.gpu_mode = config.gpu_mode or false
   self.image_dim = config.image_dim or 1024
   self.proj_dim = config.mem_dim or 300

   -- image feature embedding
   self.cell_image_emb = nn.Linear(self.image_dim, self.proj_dim)
   self.hidden_image_emb = nn.Linear(self.image_dim, self.proj_dim)

   local modules = nn.Parallel()
    :add(self.cell_image_emb)
    :add(self.hidden_image_emb)

   self.params, self.grad_params = modules:getParameters()

   -- Copy the image embedding vectors
   if config.combine_weights ~= nil then
     self.params:copy(config.combine_weights)
   end

  if gpu_mode then
    self:set_gpu_mode()
  end
end

-- Returns all of the weights of this module
function HiddenProjLayer:getWeights()
  return self.params
end

function HiddenProjLayer:getModules() 
  return {self.cell_image_emb, self.hidden_image_emb}
end

-- Sets gpu mode
function HiddenProjLayer:set_gpu_mode()
  self.cell_image_emb:cuda()
  self.hidden_image_emb:cuda()
end


-- Does a single forward step of concat layer, concatenating
-- 
function HiddenProjLayer:forward(image_feats)
   self.cell_image_proj = self.cell_image_emb:forward(image_feats)
   self.hidden_image_proj = self.hidden_image_emb:forward(image_feats)
   return {self.cell_image_proj, self.hidden_image_proj}
end

-- Does a single backward step of project layer
--
function HiddenProjLayer:backward(image_feats, cell_errors)
   -- get the image and word projection errors
   cell_image_emb_errors = cell_errors[1]
   hidden_image_emb_errors = cell_errors[2]

   -- feed them backward
   self.cell_image_emb:backward(image_feats, cell_image_emb_errors)
   self.hidden_image_emb:backward(image_feats, hidden_image_emb_errors)
end

-- Returns size of outputs of this combine module
function HiddenProjLayer:getOutputSize()
  return self.emb_dim + self.proj_dim
end

function HiddenProjLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function HiddenProjLayer:zeroGradParameters() 
  self.cell_image_emb:zeroGradParameters()
  self.hidden_image_emb:zeroGradParameters()
end

function HiddenProjLayer:normalizeGrads(batch_size)
  self.cell_image_emb.gradWeight:div(batch_size)
  self.hidden_image_emb.gradWeight:div(batch_size)
end

function HiddenProjLayer:updateParameters()
    -- normalize gradients by batch size
  self.cell_image_emb:updateParameters(self.emb_learning_rate)
  self.hidden_image_emb:updateParameters(self.emb_learning_rate)
end
