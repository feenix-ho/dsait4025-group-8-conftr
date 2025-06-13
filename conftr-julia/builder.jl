using Flux
using MLJFlux

function build_cnn(input_dim, conv_dims, mlp_dims)
    """
    Args:
        input_dim: (H, W, C)
        conv_dims: [(out_chanels, kernel_size, pool_size)]
        mlp_dims: [hidden_dim]
    Returns:


    """
    # input_dim = (H, W, C)
    H, W, C = input_dim
    layers = []
    in_ch = C
    h, w = H, W

    # Convolutional + Pooling Layers
    for (out_ch, k, pool) in conv_dims
        pad = div(k - 1, 2)
        push!(layers, Conv((k, k), in_ch => out_ch, relu, pad=pad))
        push!(layers, MaxPool((pool, pool)))
        h, w = div.(h, pool), div.(w, pool)
        in_ch = out_ch
    end

    # Flatten for Dense layers
    push!(layers, Flux.flatten)

    # Calculate flattened feature size
    feat_size = h * w * in_ch

    # Dense (MLP) Layers
    n_mlps = mlp_dims.size[1]
    in_size = feat_size
    for out_size in mlp_dims[1:n_mlps]
        push!(layers, Dense(in_size, out_size, relu))
        in_size = out_size
    end

    out_size = mlp_dims[n_mlps]
    push!(layers, Dense(in_size, out_size))
    
    return Chain(layers...)
end

function build_mlp(input_dim, mlp_dims)
    """
    Args:
        input_dim: (H, W, C)
        mlp_dims: [hidden_dim]
    """
    # input_dim = (H, W, C)
    H, W, C = input_dim
    layers = []

    # Flatten for Dense layers
    push!(layers, Flux.flatten)
    
    # Dense (MLP) Layers
    in_size = H * W * C
    n_mlps = mlp_dims.size[1]
    for out_size in mlp_dims[1:n_mlps]
        push!(layers, Dense(in_size, out_size, relu))
        in_size = out_size
    end

    out_size = mlp_dims[n_mlps]
    push!(layers, Dense(in_size, out_size))

    return Chain(layers...)
end

export build_cnn, build_mlp