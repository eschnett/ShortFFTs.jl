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

    have = [short_fft(complex(T), Tuple(input))...]
    if !isapprox(have, want; rtol=rtol)
        @show T N have want
        @show have - want
        @show maximum(abs, have - want)
    end
    @test have ≈ want rtol = rtol
end
