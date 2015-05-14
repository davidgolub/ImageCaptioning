local train_sentences = imagelstm.read_caption_sentences(train_dir, vocab)
print(train_sentences)
local train_file, err = io.open(gold_save_path,"w")

print('writing sentences to ' .. gold_save_path)
for i = 1, train_sentences.size do
  local tokens = train_sentences.sentences[i]
  local sentence = table.concat(tokens, ' ')
  print(sentence)
  train_file:write(sentence .. '\n')
end
train_file:close()