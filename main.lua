require('.')

function save_bleu(data_split)
	local data_dir = 'data/flickr8k/'
		-- load vocab
	local vocab = imagelstm.Vocab(data_dir .. 'vocab.txt')

	local train_sentences, min_size = imagelstm.read_caption_sentences(data_dir, data_split)

	gold_save_path = "predictions/bleu/gold_standard_" .. data_split .. ".ref"
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
save_bleu('test')
save_bleu('train')
save_bleu('val')



