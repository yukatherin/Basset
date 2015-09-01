BatcherX = {}
BatcherX.__index = BatcherX

function BatcherX:__init(X, batch_size)
    bat = {}
    setmetatable(bat, self)

    bat.X = X
    bat.batch_size = batch_size

    bat:reset()

    return bat
end

function BatcherX:next()
    local X_batch = nil

    if self.start <= (#self.X)[1] then
        -- allocate Tensors
        local blen = self.stop-self.start+1
        X_batch = torch.Tensor(blen, 4, 1, (#self.X)[4])

        -- copy data
        local k = 1
        for i = self.start, self.stop do
            X_batch[k] = self.X[i]
            k = k + 1
        end

        -- update batch indexes for next
        self.start = self.start + self.batch_size
        self.stop = math.min(self.stop + self.batch_size, (#self.X)[1])
    end

    return X_batch
end

function BatcherX:reset()
    self.start = 1
    self.stop = math.min(self.start+self.batch_size-1, (#self.X)[1])
end