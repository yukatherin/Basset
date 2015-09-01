#!/usr/bin/env th

require 'cutorch'
require 'cunn'
require 'lfs'

require 'batcher'
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
cmd:text()
cmd:text('Options:')
cmd:option('-seed', 1, 'RNG seed')
cmd:option('-out_dir', 'auroc', 'Dir to print AUC file and ROC points')
cmd:text()
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(opt.seed)

----------------------------------------------------------------
-- load data
----------------------------------------------------------------
local test_seqs, test_scores = load_test(opt.data_file)

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
local loss, AUCs, roc_points = convnet:test(test_seqs, test_scores)
local avg_auc = torch.mean(AUCs)

-- cd to output dir
local rs = lfs.chdir(opt.out_dir) and true or false
if not rs then
	lfs.mkdir(opt.out_dir)
	lfs.chdir(opt.out_dir)
end

-- print AUCs
local auc_out = io.open('aucs.txt', 'w')
for i=1,(#AUCs)[1] do
	auc_out:write(string.format("%-3d   %.4f\n", i, AUCs[i]))
end
auc_out:close()

print(string.format("mean  %.4f", avg_auc))
print(string.format("loss  %.3f", loss))

-- print ROC points
for yi=1,#roc_points do
	local roc_file = string.format('roc%d.txt', yi)
	local roc_out = io.open(roc_file, 'w')
	for pi = 1,(#(roc_points[yi]))[1] do
		roc_out:write(string.format("%f\t%f\n", roc_points[yi][pi][1], roc_points[yi][pi][2]))
	end
	roc_out:close()
end
