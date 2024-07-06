using CUDA
using CUDA.CUFFT
using FFTW
using ShortFFTs
using Test

# What we test
const types = [Float16, Float32, Float64, BigFloat, Complex{Float16}, Complex{Float32}, Complex{Float64}, Complex{BigFloat}]
const lengths = [collect(1:31); collect(32:8:100)]

# How to map to the types that the RNG supports
const rng_types = Dict(
    Float16 => Float16,
    Float32 => Float32,
    Float64 => Float64,
    BigFloat => Float64,
    Complex{Float16} => Complex{Float16},
    Complex{Float32} => Complex{Float32},
    Complex{Float64} => Complex{Float64},
    Complex{BigFloat} => Complex{Float64},
)

# How to map to the types that FFTW supports
const fftw_types = Dict(
    Float16 => Float32,
    Float32 => Float32,
    Float64 => Float64,
    BigFloat => Float64,
    Complex{Float16} => Complex{Float32},
    Complex{Float32} => Complex{Float32},
    Complex{Float64} => Complex{Float64},
    Complex{BigFloat} => Complex{Float64},
)

@testset "short_fft T=$T N=0" for T in filter(T -> T <: Complex, types)
    @test short_fft(T, ()) == ()
end

@testset "short_fft T=$T N=$N" for T in types, N in lengths
    RT = rng_types[T]
    input = T.(randn(RT, N))

    FT = fftw_types[T]
    want = complex(T).(fft(FT.(input)))
    rtol = max(sqrt(eps(real(T))), sqrt(eps(real(FT))))

    have = [short_fft(Tuple(input))...]
    if !isapprox(have, want; rtol=rtol)
        @show T N have want
        @show have - want
        @show maximum(abs, have - want)
    end
    @test have ≈ want rtol = rtol
end

@testset "Realistic example (CPU)" begin
    input = randn(Complex{Float32}, (32, 4))

    # Use FFTW: We need to allocate a temporary array and copy the input
    input2 = zeros(Complex{Float32}, (32, 8))
    input2[:, 1:4] = input
    want = fft(input2, 2)

    # Use ShortFFTw: We can write an efficient loop kernel
    have = Array{Complex{Float32}}(undef, (32, 8))
    for i in 1:size(input, 1)
        X = (input[i, 1], input[i, 2], input[i, 3], input[i, 4], 0, 0, 0, 0)
        Y = short_fft(X)
        for j in 1:8
            have[i, j] = Y[j]
        end
    end

    @test have ≈ want
end

if CUDA.functional()
    @inbounds function fft2!(have::CuDeviceArray{T,2}, input::CuDeviceArray{T,2}) where {T}
        i = threadIdx().x
        X = (input[i, 1], input[i, 2], input[i, 3], input[i, 4], 0.0f0, 0.0f0, 0.0f0, 0.0f0)
        Y = short_fft(X)
        for j in 1:8
            have[i, j] = Y[j]
        end
    end

    @testset "Realistic example (CUDA)" begin
        input = CUDA.randn(Complex{Float32}, (32, 4))

        # Use CUFFT: We need to allocate a temporary array and copy the input
        input2 = CUDA.zeros(Complex{Float32}, (32, 8))
        input2[:, 1:4] = input
        want = fft(input2, 2)

        # Use ShortFFTw: We can write an efficient loop kernel
        have = similar(input, (32, 8))
        @cuda threads = 32 blocks = 1 fft2!(have, input)

        @test Array(have) ≈ Array(want)
    end
end
