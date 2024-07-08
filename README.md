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

### Simple example

Here is a simple example of calling `short_fft` to transform an
arrary:

```julia
using ShortFFTs

input = randn(Complex{Float32}, (32, 8))

# We can use ShortFFT inside an efficient loop kernel
output = Array{Complex{Float32}}(undef, (32, 8))
for i in 1:32
    X = ntuple(i -> input[i, 1], 8)
    Y = short_fft(X)
    for j in 1:8
        output[i, j] = Y[j]
    end
end
```

### Non-trivial loop kernel

To show that `short_fft` can be combined with non-trivial operations
on the input data, here we first expand the input from 4 to 8 points
by setting the additional points to zero and then apply an FFT:

```julia
using ShortFFTs

input = randn(Complex{Float32}, (32, 4))

# We can use ShortFFT inside an efficient loop kernel
output = Array{Complex{Float32}}(undef, (32, 8))
for i in 1:32
    X = (input[i, 1], input[i, 2], input[i, 3], input[i, 4], 0, 0, 0, 0)
    Y = short_fft(X)
    for j in 1:8
        output[i, j] = Y[j]
    end
end
```

### Calling from CUDA

We can also call `short_fft` from CUDA (or other accelerator) code:

```julia
using CUDA
using ShortFFTs

@inbounds function fft2!(output::CuDeviceArray{T,2}, input::CuDeviceArray{T,2}) where {T}
    i = threadIdx().x
    X = (input[i, 1], input[i, 2], input[i, 3], input[i, 4], 0.0f0, 0.0f0, 0.0f0, 0.0f0)
    Y = short_fft(X)
    for j in 1:8
        output[i, j] = Y[j]
    end
end

input = CUDA.randn(Complex{Float32}, (32, 4))
output = similar(input, (32, 8))
@cuda threads=32 blocks=1 fft2!(output, input)
```

### Symbolic evaluation

Since `short_fft` is implemented purely in Julia, expressions can also
be evaluate symbolically. Notice that the input points that are known
to be zero are optimized away in the output:

```Julia
julia> using ShortFFTs
julia> using Symbolics

julia> Y = short_fft(Complex{Float64}, (x0, x1, x2, x3, 0, 0, 0, 0));

julia> foreach(1:8) do i; println("y[$(i-1)] = $(Y[i])"); end
y[0] = x0 + x1 + x2 + x3
y[1] = x0 + 0.7071067811865476x1 - 0.7071067811865476x3 + im*(-0.7071067811865476x1 - x2 - 0.7071067811865476x3)
y[2] = x0 - x2 + im*(-x1 + x3)
y[3] = x0 - 0.7071067811865476x1 + 0.7071067811865476x3 + im*(-0.7071067811865476x1 + x2 - 0.7071067811865476x3)
y[4] = x0 - x1 + x2 - x3
y[5] = x0 - 0.7071067811865476x1 + 0.7071067811865476x3 + im*(0.7071067811865476x1 - x2 + 0.7071067811865476x3)
y[6] = x0 - x2 + im*(x1 - x3)
y[7] = x0 + 0.7071067811865476x1 - 0.7071067811865476x3 + im*(0.7071067811865476x1 + x2 + 0.7071067811865476x3)
```

### Generated code

Here is the LLVM code that Julia generates for the non-trivial CPU
kernel above. (The loop also contains additional statements to load
and store data from the input and output arrays that are not shown
here.) The code consists only of floating-point additions,
subtractions, and multiplications by constants. (The value
`0x3FE6A09E60000000` is a floating-point representation of 1/√2.)
Multiplications by 1 or -1 have already been optimized away during
code generation.

```llvm
  %14 = fadd float %aggregate_load_box.sroa.0.0.copyload, 0.000000e+00
  %15 = fadd float %aggregate_load_box121.sroa.0.0.copyload, 0.000000e+00
  %16 = fadd float %14, %15
  %17 = fadd float %aggregate_load_box.sroa.3.0.copyload, %aggregate_load_box121.sroa.3.0.copyload
  %18 = fsub float %14, %15
  %19 = fsub float %aggregate_load_box.sroa.3.0.copyload, %aggregate_load_box121.sroa.3.0.copyload
  %20 = fadd float %14, %aggregate_load_box121.sroa.3.0.copyload
  %21 = fsub float %aggregate_load_box.sroa.3.0.copyload, %15
  %22 = fsub float %14, %aggregate_load_box121.sroa.3.0.copyload
  %23 = fadd float %aggregate_load_box.sroa.3.0.copyload, %15
  %24 = fadd float %aggregate_load_box75.sroa.0.0.copyload, 0.000000e+00
  %25 = fadd float %aggregate_load_box167.sroa.0.0.copyload, 0.000000e+00
  %26 = fadd float %24, %25
  %27 = fadd float %aggregate_load_box75.sroa.3.0.copyload, %aggregate_load_box167.sroa.3.0.copyload
  %28 = fsub float %24, %25
  %29 = fsub float %aggregate_load_box75.sroa.3.0.copyload, %aggregate_load_box167.sroa.3.0.copyload
  %30 = fadd float %24, %aggregate_load_box167.sroa.3.0.copyload
  %31 = fsub float %aggregate_load_box75.sroa.3.0.copyload, %25
  %32 = fsub float %24, %aggregate_load_box167.sroa.3.0.copyload
  %33 = fadd float %aggregate_load_box75.sroa.3.0.copyload, %25
  %34 = fadd float %16, %26
  %35 = fadd float %17, %27
  %36 = fsub float %16, %26
  %37 = fsub float %17, %27
  %38 = fmul float %30, 0x3FE6A09E60000000
  %39 = fmul float %31, 0x3FE6A09E60000000
  %40 = fadd float %38, %39
  %41 = fsub float %39, %38
  %42 = fadd float %20, %40
  %43 = fadd float %21, %41
  %44 = fsub float %20, %40
  %45 = fsub float %21, %41
  %46 = fadd float %18, %29
  %47 = fsub float %19, %28
  %48 = fsub float %18, %29
  %49 = fadd float %19, %28
  %50 = fmul float %33, 0x3FE6A09E60000000
  %51 = fmul float %32, 0x3FE6A09E60000000
  %52 = fsub float %50, %51
  %53 = fmul float %32, 0xBFE6A09E60000000
  %54 = fsub float %53, %50
  %55 = fadd float %22, %52
  %56 = fadd float %23, %54
  %57 = fsub float %22, %52
  %58 = fsub float %23, %54
```
