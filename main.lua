require('.')

function save_bleu_flickr8k(data_dir, data_split, save_data_split)
	
	local train_sentences, min_size = 
		imagelstm.read_caption_sentences(data_dir, data_split)

	gold_save_path = "predictions/bleu/flickr8k/gold_standard_" .. save_data_split .. ".ref"
	train_files = {}
	for i = 1, min_size do
		local train_file, err = io.open(gold_save_path .. i - 1, "w")
		table.insert(train_files, train_file)
	end

	print('writing sentences to ' .. gold_save_path .. " " .. train_sentences.size)
	for i = 1, train_sentences.size do
	  local sentences = train_sentences.sentences[i]
	  for j = 1, min_size do
	  	local tokens = sentences[j]['tokens']
	  	local sentence = table.concat(tokens, ' ')
	  	train_files[j]:write(sentence)
	  	if i < train_sentences.size then 
	  		train_files[j]:write('\n')
	  	end
	  end
	end

	for i = 1, #train_files do
		train_files[i]:close()
	end
end

function save_bleu_coco(data_dir, data_split, save_data_split)
	
		-- load vocab
	local train_sentences, min_size = 
		imagelstm.read_caption_sentences(data_dir .. data_split .. '/', 'train')

	gold_save_path = "predictions/bleu/coco/gold_standard_" .. data_split .. ".ref"
	train_files = {}
	for i = 1, min_size do
		local train_file, err = io.open(gold_save_path .. i - 1, "w")
		table.insert(train_files, train_file)
	end

	print('writing sentences to ' .. gold_save_path .. " " .. train_sentences.size)
	for i = 1, train_sentences.size do
	  local sentences = train_sentences.sentences[i]
	  for j = 1, min_size do
	  	local tokens = sentences[j]['tokens']
	  	local sentence = table.concat(tokens, ' ')
	  	train_files[j]:write(sentence)
	  	if i < train_sentences.size then 
	  		train_files[j]:write('\n')
	  	end
	  end
	end

	for i = 1, #train_files do
		train_files[i]:close()
	end
end

-- directory containing dataset files
save_bleu_flickr8k('data/flickr8k/', 'test', 'test')
save_bleu_flickr8k('data/flickr8k/', 'train', 'train')
save_bleu_flickr8k('data/flickr8k/', 'val', 'val')

-- directory containing coco files
save_bleu_coco('data/coco/', 'train', 'train')
save_bleu_coco('data/coco/', 'test', 'val')
save_bleu_coco('data/coco/', 'test', 'test')



