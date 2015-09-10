#!/usr/bin/env th

require 'convnet'
require 'convnet_io'
require 'postprocess'

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
cmd:option('-norm', false, 'Normalize all targets to a level plane')
cmd:text()
opt = cmd:parse(arg)

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

----------------------------------------------------------------
-- predict and test
----------------------------------------------------------------
-- guarantee evaluate mode
convnet.model:evaluate()

-- measure accuracy on a test set
local preds = convnet:predict(test_seqs)

if opt.norm then
    -- TEMP! TEMP! TEMP!
    if convnet.pred_means == nil then
        convnet.pred_means = preds:mean(1):squeeze()
    end

    preds = troy_norm(preds, convnet.pred_means)
end

----------------------------------------------------------------
-- dump to file
----------------------------------------------------------------
local predict_out = io.open(opt.out_file, 'w')

for si=1,(#preds)[1] do
    predict_out:write(preds[{si,1}])
    for ti=2,(#preds)[2] do
        predict_out:write("\t",preds[{si,ti}])
    end
    predict_out:write("\n")
end

predict_out:close()

-- local hdf_out = hdf5.open(opt.out_file, 'w')
-- hdf_out:write("preds", preds)
-- hdf_out:close()
