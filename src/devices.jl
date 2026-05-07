device(v::T) where T <: CuArray = gpu
device(v::T) where T <: AbstractArray = cpu
device(v::SubArray{T,N,P,I,L}) where {T, N, P <: CuArray, I, L} = gpu
device(v::SubArray{T,N,P,I,L}) where {T, N, P <: AbstractArray, I, L} = cpu
function device(c)
    # Flux 0.16 port: implicit `Flux.params` is gone. Walk the model tree with
    # Functors; if any CuArray leaf is present we live on the GPU, otherwise CPU.
    is_gpu = Ref(false)
    Flux.fmap(c) do x
        if x isa CuArray
            is_gpu[] = true
        end
        x
    end
    is_gpu[] ? gpu : cpu
end

# Call F with input x but ensure they are both on the device of F
gpucall(F, x::CuArray) = F(x)
gpucall(F, x::SubArray{T,N,P,I,L}) where {T, N, P <: CuArray, I, L} = F(x)

gpucall(F, x::AbstractArray) = cpu(F(gpu(x)))

cpucall(F, x::AbstractArray) = F(x)

cpucall(F, x::CuArray) = gpu(F(cpu(x)))
cpucall(F, x::SubArray{T,N,P,I,L}) where {T, N, P <: CuArray, I, L} = gpu(F(cpu(x)))

mdcall(F, x, device) = device == gpu ? gpucall(F,x) : cpucall(F, x)

@inline function bslice(v, i)
    nd = ndims(v)
    if nd == 2
        return view(v,:,i)
    elseif nd == 3
        return view(v, :, :, i)
    elseif nd == 4
        return view(v, :, :, :, i)
    else
        return view(v, ntuple(x->:, nd-1)..., i)
    end
end
