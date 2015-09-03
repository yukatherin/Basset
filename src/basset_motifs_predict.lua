#!/usr/bin/env th

require 'cunn'
require 'cutorch'
require 'hdf5'

require 'batcher'
require 'convnet_io'
require 'convnet'

----------------------------------------------------------------
-- parse arguments
----------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('DNA ConvNet hidden layer visualizations')
cmd:text()
cmd:text('Arguments')
cmd:argument('model_file')
cmd:argument('data_file')
cmd:argument('out_file')
cmd:text()
cmd:text('Options:')
cmd:option('-sample', 1000, 'Sample sequences to compute on')
cmd:text()
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(1)

----------------------------------------------------------------
-- load data
----------------------------------------------------------------
local convnet_params = torch.load(opt.model_file)
local convnet = ConvNet:__init()
convnet:load(convnet_params)
convnet:decuda()

-- get convolution filter stats
local conv_filters = convnet.conv_filters[1]
local filter_size = convnet.conv_filter_sizes[1]

local test_seqs = load_test_seqs(opt.data_file)

-- down sample
local batcher = BatcherX:__init(test_seqs, opt.sample)
local X = batcher:next()

----------------------------------------------------------------
-- predict
----------------------------------------------------------------
-- predict seqs
convnet.model:evaluate()
convnet:predict(X, opt.sample)

-- get convolution filter params
local filter_weights = convnet.model.modules[1].weight:squeeze()

-- get convolution filter outputs
local filter_outs = convnet:get_nonlinearity(1).output:squeeze()

-- dump to file, load into python.
local hdf_out = hdf5.open(opt.out_file, 'w')
hdf_out:write('weights', filter_weights)
hdf_out:write('outs', filter_outs)
hdf_out:close()