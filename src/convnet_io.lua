require 'hdf5'
require 'torch'

-------------------------------------------------------------------------
-- load_test
--
-- Load the datasets labeled 'train' and 'valid'
--------------------------------------------------------------------------
function load_data(hdf5_file, add_rc)
	local myFile = hdf5.open(hdf5_file, 'r')
	
	local train_scores = tensor_raw(myFile:read('train_out'):all())
	local train_seqs_raw = myFile:read('train_in'):all()
	local valid_scores = tensor_raw(myFile:read('valid_out'):all())
	local valid_seqs_raw = myFile:read('valid_in'):all()

	myFile:close()

	local seq_len = (#train_seqs_raw)[2]/4

	local train_seqs = torch.reshape(train_seqs_raw, (#train_seqs_raw)[1], 4, 1, seq_len)
	local valid_seqs = torch.reshape(valid_seqs_raw, (#valid_seqs_raw)[1], 4, 1, seq_len)

	if add_rc then
		train_seqs, train_scores = cat_rc(train_seqs, train_scores)
		valid_seqs, valid_scores = cat_rc(valid_seqs, valid_scores)
	end

	collectgarbage()

	return train_seqs, train_scores, valid_seqs, valid_scores
end

--------------------------------------------------------------------------
-- load_test
--
-- Load just one dataset labeled 'test'
--------------------------------------------------------------------------
function load_test(hdf5_file)
	local myFile = hdf5.open(hdf5_file, 'r')
	
	local test_scores = tensor_raw(myFile:read('test_out'):all())
	local test_seqs_raw = myFile:read('test_in'):all()

	myFile:close()

	local test_seqs = torch.reshape(test_seqs_raw, (#test_seqs_raw)[1], 4, 1, (#test_seqs_raw)[2]/4)
	
	return test_seqs, test_scores
end

--------------------------------------------------------------------------
-- load_test_seqs
--
-- Load sequences only from one dataset labeled 'test'
--------------------------------------------------------------------------
function load_test_seqs(hdf5_file)
	local myFile = hdf5.open(hdf5_file, 'r')
	local test_seqs_raw = myFile:read('test_in'):all()
	myFile:close()

	local test_seqs = torch.reshape(test_seqs_raw, (#test_seqs_raw)[1], 4, 1, (#test_seqs_raw)[2]/4)
	
	return test_seqs
end


--------------------------------------------------------------------------
-- tensor_raw
--
-- I'm not sure what this is doing to be honest. I must've thought I
-- was changing the type, and maybe I am. Dunno.
--------------------------------------------------------------------------
function tensor_raw(raw_data)
	local tensor_data = torch.Tensor(#raw_data)
	for i = 1,(#raw_data)[1] do
		for j = 1,(#raw_data)[2] do
			tensor_data[{i,j}] = raw_data[{i,j}]
		end
	end
	return tensor_data
end


--------------------------------------------------------------------------
-- cat_rc
--
-- Supplement the given sequences and scores with reverse complemented
-- sequences and duplicated scores.
--------------------------------------------------------------------------
function cat_rc(seqs, scores)
	local num_seqs = (#seqs)[1]
	local seq_len = (#seqs)[4]
	local seqs_rc = seqs:clone()

	-- reverse
	for si=1,num_seqs do
		for ni=1,4 do
			for pos=1,seq_len do
				seqs_rc[{si,ni,1,pos}] = seqs[{si,ni,1,seq_len-pos+1}]
			end
		end
	end

	-- complement
	for si=1,num_seqs do
		-- save A
		local A_tmp = seqs_rc[{si,1,1,{}}]

		-- set A to T
		seqs_rc[{si,1,1,{}}] = seqs_rc[{si,4,1,{}}]

		-- set T to saved A
		seqs_rc[{si,4,1,{}}] = A_tmp

		-- save C
		local C_tmp = seqs_rc[{si,2,1,{}}]

		-- set C to G
		seqs_rc[{si,2,1,{}}] = seqs_rc[{si,3,1,{}}]

		-- set G to saved C
		seqs_rc[{si,3,1,{}}] = C_tmp
	end

	-- cat seqs
	seqs = torch.cat(seqs, seqs_rc, 1)

	-- duplicate scores
	local scores_rc = scores:clone()

	-- cat scores
	scores = torch.cat(scores, scores_rc, 1)

	return seqs, scores
end


--------------------------------------------------------------------------
-- get_1hot
--
-- Return the nucleotide at position pos in the 4x1xlen Tensor seq_1hot.
--------------------------------------------------------------------------
function get_1hot(seq_1hot, pos)
	if seq_1hot[{1,1,pos}] == 1 then
		return 'A'
	elseif seq_1hot[{2,1,pos}] == 1 then
		return 'C'
	elseif seq_1hot[{3,1,pos}] == 1 then
		return 'G'
	elseif seq_1hot[{4,1,pos}] == 1 then
		return 'T'
	else
		return 'N'
	end
end


--------------------------------------------------------------------------
-- set_1hot
--
-- Set position pos in the 4x1xlen Tensor seq_1hot to represent nt.
--------------------------------------------------------------------------
function set_1hot(seq_1hot, pos, nt)
	-- zero table
	for i=1,4 do
		seq_1hot[{i,1,pos}] = 0
	end

	if nt == 'A' then
		seq_1hot[{1,1,pos}] = 1
	elseif nt == 'C' then
		seq_1hot[{2,1,pos}] = 1
	elseif nt == 'G' then
		seq_1hot[{3,1,pos}] = 1
	elseif nt == 'T' then
		seq_1hot[{4,1,pos}] = 1
	else
		print("Unrecognized nt", nt)
	end
end
