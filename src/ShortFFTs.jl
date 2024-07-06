module ShortFFTs

using Primes

export short_fft

# Cooley-Tukey <https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm>

# Unzip a vector of tuples
unzip(xs::AbstractVector{<:Tuple}) = ([x[1] for x in xs], [x[2] for x in xs])

# Find a common type for a tuple
promote_tuple(X::Tuple) = promote_type((typeof(x) for x in X)...)

cw(x) = Complex(imag(x), -real(x)) # -im * x

# Apply a phase to an expression
function phase(::Type{T}, k::Rational, expr) where {T<:Complex}
    k = mod(k, 1)
    # Ensure that simple cases do not lead to arithmetic operations
    k == 0 && return expr
    k >= 1//2 && return phase(T, k - 1//2, :(-$expr))
    k >= 1//4 && return phase(T, k - 1//4, :(cw($expr)))
    k == 1//8 && return :($((1 - im) * sqrt(T(1//2))) * $expr)
    # Prevent round-off by evaluating the phase constants with very high precision
    return :($(T(cispi(BigFloat(-2 * k)))) * $expr)
end
phase(::Type{T}, k::Integer, expr) where {T<:Complex} = phase(T, Rational(k), expr)

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
        # term(i, n) = :($(phase(T, ((i - 1) * (n - 1))//N)) * $(X[i]))
        term(i, n) = phase(T, ((i - 1) * (n - 1))//N, X[i])
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
    # twiddle(n1, n2) = phase(T, ((n1 - 1) * (n2 - 1))//(N1 * N2))
    twiddle(n1, n2, expr) = phase(T, ((n1 - 1) * (n2 - 1))//(N1 * N2), expr)
    # Second step: N2 FFTs of size N1
    codeZs, Zs = unzip([gen_fft(T, [twiddle(n1, n2, Ys[n1][n2]) for n1 in 1:N1]) for n2 in 1:N2])
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
        begin
            $code
        end
        return tuple($(res...))
    end
end

@inline short_fft(X::Tuple) = short_fft(complex(promote_tuple(X)), X)

@inline short_fft(X::NTuple{N,T}) where {N,T<:Complex} = short_fft(T, X)
short_fft(X::AbstractVector{T}) where {T<:Complex} = short_fft(T, Tuple(X))

@inline short_fft(X::NTuple{N,T}) where {N,T<:Real} = short_fft(complex(T), X)
short_fft(X::AbstractVector{T}) where {T<:Real} = short_fft(complex(T), Tuple(X))

end
