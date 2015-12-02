#!/usr/bin/env th

cuda = true
require 'convnet'

----------------------------------------------------------------
-- parse arguments
----------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Move a DNA ConvNet off CUDA to the CPU')
cmd:text()
cmd:text('Arguments')
cmd:argument('model_file')
cmd:argument('out_file')
cmd:text()
opt = cmd:parse(arg)

----------------------------------------------------------------

-- load parameters
local convnet_params = torch.load(opt.model_file)

-- construct network
local convnet = ConvNet:__init()
convnet:load(convnet_params)

-- move off GPU
convnet:decuda()

-- save back to disk
torch.save(opt.out_file, convnet)
