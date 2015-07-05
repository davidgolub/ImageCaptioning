--[[

  Math helper functions on tables

--]]

-- Check type of input
function check_type(input, desired_type)
  local input_type = torch.typename(input)
  assert(input_type == desired_type, "input has type " .. input_type ..
   " but desired is " .. desired_type)
end

-- Enable dropouts
function enable_sequential_dropouts(model)
   for i,m in ipairs(model.modules) do
      if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
        m:training()
      end
   end
end

-- Disable dropouts
function disable_sequential_dropouts(model)
   for i,m in ipairs(model.modules) do
      if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
        m:evaluate()
      end
   end
end

-- Convert 1-d torch tensor to lua table
function tensor_to_array(t1)
  -- This assumes `t1` is a 2-dimensional tensor!
  local t2 = {}
  for i=1,t1:size(1) do
    t2[i] = t1[i]
  end
  return t2
end


-- Sorts tables by first value
-- first_entry, second_entry are tables
function min_sort_function(first_table, second_table)
    return first_table[1] < second_table[1]
end


-- Sorts tables by first value
-- first_entry, second_entry are tables
function max_sort_function(first_table, second_table)
    return first_table[1] > second_table[1]
end

-- Argmax: hacky way to ignore end token to reduce silly sentences
function argmax(v, ignore_end)
  local idx = 1
  local max = v[1]
  local start_index = 2
  if ignore_end then
    start_index = 4
  end

  for i = start_index, v:size(1) do
    if v[i] > max then
      max = v[i]
      idx = i
    end
  end
  return idx
end

-- TopkArgmax returns top k indices, values from list
function topk(list, k)
  tmp_list = {}
  for i = 1, #list do
    table.insert(tmp_list, list[i])
  end
  table.sort(tmp_list, max_sort_function)

  max_entries = {}
  for i = 1, k do
    table.insert(max_entries, tmp_list[i])
  end

  return max_entries
end

-- TopkArgmax returns top k indices, values from list
function topkargmax(list, k)
  tmp_list = {}
  for i = 1, list:size(1) do
    table.insert(tmp_list, {list[i], i})
  end
  table.sort(tmp_list, max_sort_function)

  max_indices = {}
  for i = 1, k do
    table.insert(max_indices, tmp_list[i][2])
  end
  return max_indices
end
