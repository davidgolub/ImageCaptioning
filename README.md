Image Captioning using LSTMs
===============================================

An implementation of the Image Captioning LSTM architectures in torch.
## Requirements

- [Torch7](https://github.com/torch/torch7)
- [penlight](https://github.com/stevedonovan/Penlight)
- [nn](https://github.com/torch/nn)
- [nngraph](https://github.com/torch/nngraph)
- [optim](https://github.com/torch/optim)
- [cutorch] for torch gpu support
- [cunn] for nn gpu support
- Java >= 8 (for Stanford CoreNLP utilities)
- Python >= 2.7

The Torch/Lua dependencies can be installed using [luarocks](http://luarocks.org). For example:

```
luarocks install nngraph
```

To run the program do:
```
./fetch_and_preprocess.sh
./setup_data.sh
th image_captioning/main.lua
```
