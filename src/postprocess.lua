-- Troyanskaya's normalization
function troy_norm(p, u)
    local pn = torch.log(torch.cdiv(p,(-p+1)))
    local xn = math.log(0.05/0.95)
    local un = torch.log(torch.cdiv(u,-u+1)):repeatTensor((#p)[1], 1)
    local zn = pn + xn - un
    return (torch.exp(-zn)+1):pow(-1)
end
