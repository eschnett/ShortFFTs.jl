# ShortFFTs.jl

Efficient and inlineable short Fast Fourier Transforms

* [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://eschnett.github.io/ShortFFTs.jl/dev/)
* [![GitHub
  CI](https://github.com/eschnett/ShortFFTs.jl/workflows/CI/badge.svg)](https://github.com/eschnett/ShortFFTs.jl/actions)
* [![codecov](https://codecov.io/gh/eschnett/ShortFFTs.jl/branch/main/graph/badge.svg?token=75FT03ULHD)](https://codecov.io/gh/eschnett/ShortFFTs.jl)
* [![PkgEval](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/S/ShortFFTs.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/S/ShortFFTs.html)

The `ShortFFTs.jl` package provides a single function, `short_fft`,
that performs a Fast Fourier Transform (FFT) on its input. Different
from the `fft` functions provided via the `AbstractFFTs.jl` interface
(e.g. by `FFTW.jl`), `short_fft` is designed to be called for a single
transform at a time.

One major advantage of `short_fft` is that it can be efficiently
called from within a loop kernel, or from device code (e.g. a CUDA
kernel). This allows combining several operations into a single loop,
e.g. pre-processing input data by applying a window function.

The term "short" in the name refers to FFT lengths less than about
1000.

## API

The function `short_fft` expects its input as a tuple. One also needs
to specify the output element type. The output of the FFT is returned
as a tuple as well. This makes sense because the call to `short_fft`
is expected to be inlined, and using arrays would introduce an
overhead.

```julia
using ShortFFTs

# short_fft(::Type{T<:Complex}, input::Tuple)::Tuple
output = short_fft(T, input)

# short_fft(input::Tuple)::Tuple
output = short_fft(input)

# short_fft(input::AbstractVector)::Tuple
output = short_fft(input)
```

`short_fft` can be called by device code, e.g. in a CUDA kernel.

`short_fft` is a "generated function" that is auto-generated at run
time for each output type and tuple length. The first call to
`short_fft` for a particular combination of argument types is much
more expensive than the following because Julia needs to generate the
optimzed code.

Since `short_fft` is implemented in pure Julia it can work with any
(complex) type. The implementation uses local variables as scratch
storage, and very large transforms are thus likely to be inefficient.

## Algorithm

`ShortFFTs.jl` implements the [Cooley-Tukey FFT
algorithm](https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm)
as described on Wikipedia. It has been tested against `FFTW.jl`.

Transforms of any lenghts are supported. Lengths without large prime
factors (especially powers of two) are significantly more efficient.
`256` is a very good length, `257` is a very bad one (since it is a
prime number).

`short_fft` is not parallelized at all. It is expected that the
surrounding code is parallelized if that makes sense. For long
transforms `AbstractFFTs.jl` will provide a better performance.

## Example

This example first expands its input from 4 to 8 points by setting the
additional points to zero and then applies an FFT.

```julia
input = randn(Complex{Float32}, (32, 4))

# Use ShortFFTw: We can write an efficient loop kernel
output = Array{Complex{Float32}}(undef, (32, 8))
for i in 1:32
    X = (input[i, 1], input[i, 2], input[i, 3], input[i, 4], 0, 0, 0, 0)
    Y = short_fft(X)
    for j in 1:8
        output[i, j] = Y[j]
    end
end
```
