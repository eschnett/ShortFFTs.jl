module ShortFFTs

using Primes

export short_fft

# Cooley-Tukey <https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm>

# Unzip a vector of tuples
unzip(xs::AbstractVector{<:Tuple}) = ([x[1] for x in xs], [x[2] for x in xs])

# Calculate phase
function phase1(::Type{T}, k::Rational) where {T<:Complex}
    k = mod(k, 1)
    k == 0 && return T(1)
    k >= 1//2 && return -phase1(T, k - 1//2)
    k >= 1//4 && return -im * phase1(T, k - 1//4)
    k == 1//8 && return (1 - im) * sqrt(T(1//2))
    # k > 1//8 && return (1 - im) * sqrt(T(1//2)) * phase1(T, k - 1//8)
    return cispi(-2 * T(k))
end
phase(::Type{T}, k::Rational) where {T<:Complex} = T(phase1(Complex{BigFloat}, k))
phase(::Type{T}, k::Integer) where {T<:Complex} = phase(T, Rational(k))
@assert all(phase(Complex{Float64}, i//N) ≈ cispi(-2 * i / N) for N in 1:17 for i in 0:(2 * N))

# Generate a short FFT
function gen_fft(::Type{T}, X::AbstractVector) where {T<:Complex}
    N = length(X)

    # Trivial case
    if N == 0
        code = quote end
        res = []
        return code, res
    end

    # Base case
    if N == 1
        Y = gensym(:Y)
        code = quote
            $Y = T($(X[1]))
        end
        res = [Y]
        return code, res
    end

    # Handle prime lengths directly
    if isprime(N)
        Y = [gensym(Symbol(:Y, n - 1)) for n in 1:N]
        term(i, n) = :($(phase(T, ((i - 1) * (n - 1))//N)) * $(X[i]))
        stmts = [
            quote
                $(Y[n]) = +($([term(i, n) for i in 1:N]...))
            end for n in 1:N
        ]
        code = quote
            $(stmts...)
        end
        return code, Y
    end

    # TODO: Use split-radix FFT for N % 4 = 0 <https://en.wikipedia.org/wiki/Split-radix_FFT_algorithm>

    # Apply Cooley-Tukey <https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm> with the smallest prime factor
    (N1, _), _ = iterate(eachfactor(N))
    @assert N % N1 == 0
    N2 = N ÷ N1
    # First step: N1 FFTs of size N2
    codeYs, Ys = unzip([gen_fft(T, [X[i] for i in n1:N1:N]) for n1 in 1:N1])
    # twiddle factors
    twiddle(n1, n2) = phase(T, ((n1 - 1) * (n2 - 1))//(N1 * N2))
    # Second step: N2 FFTs of size N1
    codeZs, Zs = unzip([gen_fft(T, [:($(twiddle(n1, n2)) * $(Ys[n1][n2])) for n1 in 1:N1]) for n2 in 1:N2])
    # Combine results
    code = quote
        $(codeYs...)
        $(codeZs...)
    end
    Z = [Zs[n2][n1] for n1 in 1:N1 for n2 in 1:N2]
    return code, Z
end

@generated function short_fft(::Type{T}, X::Tuple) where {T<:Complex}
    N = length(fieldnames(X))
    code, res = gen_fft(T, [:(X[$n]) for n in 1:N])
    return quote
        Base.@_inline_meta
        @fastmath begin
            $code
        end
        return tuple($(res...))::NTuple{$N,$T}
    end
end

short_fft(X::NTuple{N,T}) where {N,T<:Complex} = short_fft(T, X)
short_fft(X::AbstractVector{T}) where {T<:Complex} = short_fft(T, Tuple(X))

short_fft(X::NTuple{N,T}) where {N,T<:Real} = short_fft(complex(T), X)
short_fft(X::AbstractVector{T}) where {T<:Real} = short_fft(complex(T), Tuple(X))

end
