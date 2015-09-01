#!/usr/bin/env th

require 'cutorch'
require 'cunn'

require 'convnet'
require 'convnet_io'

local access_token = "2b0784a1-996d-4aed-8f2f-058b8325aa6a"

----------------------------------------------------------------
-- parse arguments
----------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('DNA ConvNet testing')
cmd:text()
cmd:text('Arguments')
cmd:argument('model_file')
cmd:argument('data_file')
cmd:argument('out_file')
cmd:text()
cmd:text('Options:')
cmd:option('-seed', 1, 'RNG seed')
cmd:text()
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(opt.seed)

----------------------------------------------------------------
-- load data
----------------------------------------------------------------
local test_seqs = load_test_seqs(opt.data_file)

----------------------------------------------------------------
-- construct model
----------------------------------------------------------------
-- initialize
local convnet = ConvNet:__init()

-- load from saved parameters
local convnet_params = torch.load(opt.model_file)
convnet:load(convnet_params)
convnet:decuda()

----------------------------------------------------------------
-- predict and test
----------------------------------------------------------------
-- guarantee evaluate mode
convnet.model:evaluate()

-- measure accuracy on a test set
local preds = convnet:predict(test_seqs)

----------------------------------------------------------------
-- dump to file
----------------------------------------------------------------
local hdf_out = hdf5.open(opt.out_file, 'w')
hdf_out:write("preds", preds)
hdf_out:close()
