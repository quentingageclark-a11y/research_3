### A Pluto.jl notebook ###
# v0.20.27

#> [frontmatter]
#> chapter = 4
#> section = 6
#> order = 6
#> image = "https://raw.githubusercontent.com/fonsp/Pluto.jl/580ab811f13d565cc81ebfa70ed36c84b125f55d/demo/plutodemo.gif"
#> title = "Fourier Transform"
#> tags = ["module4", "track_julia", "track_material", "Pluto", "PlutoUI"]
#> layout = "layout.jlhtml"

using Markdown
using InteractiveUtils

# ╔═╡ 37dbf60a-86ab-4bd3-bf70-4362a1054df6
begin
	using PlutoUIExtra
	TableOfContents()
end

# ╔═╡ d287df06-9393-4817-b0a2-b687ec3d1997
using Plots

# ╔═╡ b697c05d-5bea-44ec-a88b-bc6df24fd7fa
using FFTW

# ╔═╡ 889e94b1-7e20-4ba0-85bf-31023d0090d3
using PyCall

# ╔═╡ de7ddbcb-4417-46a3-ba8c-a0f9f30f1915
using Conda

# ╔═╡ 17b5cd2e-284a-4dd3-bc28-5d2f815ff3b7
using BenchmarkTools, DataFrames

# ╔═╡ 42bf53fb-6164-4187-ae78-a6a6c859a1b7
using Statistics, LaTeXStrings

# ╔═╡ ed8e7bfa-ccc9-423f-838a-94ae025326ce
using Polynomials

# ╔═╡ d1097cf8-e5e5-4902-b3d4-387ad103fbb0
using DelimitedFiles

# ╔═╡ a295efe1-1cce-4fd6-a6d4-039765cd86de
import HTTP

# ╔═╡ f2492ddd-d71a-4d1a-932c-a314de792f93
let
    notebook_dir = @__DIR__
    toc_path = joinpath(notebook_dir, "TOC_Notebook.jl")
    
    Markdown.parse("[⬅️ Back to Table of Contents](./open?path=$(HTTP.escapeuri(toc_path)))")
end

# ╔═╡ 0ddbb258-0d65-4541-8ea7-37f4220bc934
md"""
# Fourier transform


The Fourier transform is one of the most essential tools in data analysis. It decomposes a signal into harmonics (complex exponentials). It is useful both as a computational tool and as an example of signal decomposition, analogous to other transforms.
"""

# ╔═╡ 64618f52-c56d-4cc5-89f4-6f8b5348b290
md"""
## Continuous versus discrete

The continuous Fourier transform is defined as a transformation from a time-domain function $f(t)$ to its frequency-domain counterpart $F(\omega)$:

$$F(\omega) = \int\limits_{-\infty}^{\infty} f(t)\,e^{-i\omega\,t}\,dt$$
"""

# ╔═╡ 2f2ec598-9e50-4f25-8179-b118c39e783c
md"""
with the inverse transform given by

$$f(t) = \frac{1}{2\pi}\,\int\limits_{-\infty}^{\infty} F(\omega)\,e^{i\omega\,t}\,d\omega\;.$$

The choice of the sign in the exponential ($+$ or $-$) and the placement of the normalization factor $1/(2\pi)$ between the forward and the inverse transform are arbitrary; different scientific fields may use other conventions.
"""

# ╔═╡ ef185e26-899c-4ebc-bf82-153c1e5d6223
md"""
Suppose that our continuous signal $f(t)$ is a sequence of $N$ samples with a regular sampling interval $\Delta t$:

$$f(t)=\sum\limits_{n=0}^{N-1} f_n\delta(t-n\,\Delta t)\;,$$

where $\delta$ is Dirac's *delta function*. 
"""

# ╔═╡ 047cd9b1-2800-49a3-9350-ae003f37902c
md"""
Applying the continuous Fourier transform, we obtain

$$\begin{array}{rcl}F(\omega) & = & \displaystyle \int \left[\sum\limits_{n=0}^{N-1} f_n\,\delta(t-n\,\Delta t)\right] \,e^{-i\omega\,t}\,dt 
\\ & = &
          \displaystyle\sum\limits_{n=0}^{N-1} f_n\,\left[\int \delta(t-n\,\Delta t)\,e^{-i\omega\,t}\,dt\right] \\ & = & 
          \displaystyle\sum\limits_{n=0}^{N-1} f_n\,e^{-i\omega\,n\,\Delta t}\;.\end{array}$$
"""

# ╔═╡ 47d4f2bd-1580-40e1-bac2-5f527a2ef7fc
md"""
The complex exponential function $e^{-i\omega\,n\,\Delta t}$ is periodic and represents rotations around the unit circle in the complex plane. Correspondingly, the Fourier transform $F(\omega)$ becomes periodic as well. If we want to sample it with $N$ samples, we can choose the samples to be equally spaced around the circle, as follows:

$$\omega_k = \frac{2\pi\,k}{N\,\Delta t}\;,\quad k=0,1,2,\cdots,N-1\;.$$

The sampled (discrete) Fourier transform is then

$$F_k = \sum\limits_{n=0}^{N-1} f_n\,e^{-i\omega_k\,n\,\Delta t} = \sum\limits_{n=0}^{N-1} f_n\,e^{-i\,2\,\pi\,k\,n/N}\;.$$
"""

# ╔═╡ 2119a94f-f033-4e94-96de-d5caa4048cef
N = 128

# ╔═╡ 9b700931-4446-4758-b211-745bb2dc3429
begin
	plots = Dict()
	for (name, func) in Dict("real" => cos, "imaginary" => sin)
	    matrix = [func(2π/N*(k-1)*(n-1)) for k in 1:N, n in 1:N]
	    plots[name] = heatmap(matrix, 
						 title="Fourier matrix: $(name)", 
	                     color=:grays, legend=:none,
						 aspect_ratio=:equal, 
	                     xlim=(0.5,N+0.5), ylim=(0.5,N+0.5))
	end
end

# ╔═╡ c14c9130-1186-48df-b7c6-6424000c17ba
plot(plots["real"], plots["imaginary"], layout=(1, 2))

# ╔═╡ da8af9ef-7baf-4602-9dcc-e87c091b10a7
md"""
The adjoint of the discrete Fourier transform is a multiplication by the complex-conjugate transpose of $A_{kn}$:

$$g_n = \sum\limits_{k=0}^{N-1} F_k\,e^{i\,2\,\pi\,k\,n/N}$$

or substituting the forward transform,

$$g_n = \sum\limits_{k=0}^{N-1} \sum\limits_{m=0}^{N-1} f_m\,e^{i\,\frac{2\,\pi\,(n-m)\,k}{N}} = 
    \sum\limits_{m=0}^{N-1} f_m \left( \sum\limits_{k=0}^{N-1} e^{i\,\frac{2\,\pi\,(n-m)\,k}{N}}\right)\;.$$
"""

# ╔═╡ 73fed2f1-5075-43f4-86f5-f6bc7ad778a8
md"""
Let $W=e^{i\,\frac{2\,\pi\,(n-m)}{N}}$ and note that $W^N = e^{i\,2\,\pi\,(n-m)} = 1$ and, therefore,

$$\begin{array}{rcl}\sum\limits_{k=0}^{N-1} e^{i\,\frac{2\,\pi\,(n-m)\,k}{N}} & = & \sum\limits_{k=0}^{N-1} W^k \\ & = & 1 + W + W^2 + \cdots + W^{N-1} \\ & = & 
\left\{\begin{array}{ll}
\displaystyle N\;,\quad & n=m\;. \\
\displaystyle \frac{1-W^N}{1-W}=0\;,\quad & n \ne m\;.
\end{array}\right.\end{array}$$
"""

# ╔═╡ be83061d-c0a5-4091-bd20-3b62d3f8510f
md"""
Substituting it into the equation for the adjoint leads to $g_n = N\,f_n$. Thus, the adjoint of the Fourier transform differs from the inverse only by the scaling factor of $N$. The inverse discrete Fourier transform is therefore

$$f_n = \frac{1}{N}\,\sum\limits_{k=0}^{N-1} F_k\,e^{i\,2\,\pi\,k\,n/N}\;.$$

* **Bracewell, R., 1999, The Fourier transform and its applications, 3rd ed.: McGraw-Hill.**
"""

# ╔═╡ d8c93879-0785-42cc-b110-f32b8db6964c
md"""
## Why use the Fourier transform?

The Fourier transform has notable properties that make it valuable for data analysis. We will focus on two: the shift property and the derivative property.
"""

# ╔═╡ e83191d8-9bfc-4222-a86b-65e07667d448
md"""
### Shift property

In the continuous Fourier transform, if the input signal $f(t)$ is
shifted by $\Delta t$, its transform becomes

$$\begin{array}{rcl}\int f(t-\Delta t)\,e^{-i\omega\,t}\,dt & = & e^{i\omega\,\Delta t}\,\int f(t-\Delta t)\,e^{-i\omega\,(t-\Delta t)}\,dt \\ & = & 
        e^{-i\omega\,\Delta t}\,F(\omega)\;.\end{array}$$

Therefore, a shift corresponds to multiplication by a complex exponential in the Fourier domain.
"""

# ╔═╡ eb8e5c8d-cfda-4010-b666-7ec41ee06902
md"""
A similar property exists for the discrete Fourier transform. Shifting of a periodic signal by one sample results in multiplication by
$e^{-i\,\frac{2\,\pi\,k}{N}}$. Therefore, we can write

$$Z_k=e^{-i\,2\,\pi\,k/N} = e^{-i\,\omega_k\,\Delta t}\;,$$

and conclude that multiplication by $Z_k$ has the same meaning as multiplication by $Z$ in the Z-transform notation we used previously to describe digital convolution. 
"""

# ╔═╡ b549c4a2-edc3-4837-972b-d56c003ed6ae
md"""
With that definition, the digital Fourier transform is equivalent to the polynomial

$$F_k = \sum\limits_{n=0}^{N-1} f_n\,Z_k^n=F(Z_k)\;,$$

and multiplication of polynomials (digital convolution) corresponds to point-by-point multiplication of Fourier transforms. In other words, the digital Fourier transform *diagonalizes* the convolution matrix.
"""

# ╔═╡ 7e1baaef-e73a-49e2-baed-e1e620ff0712
md"""
This observation provides an alternative way of implementing convolution. Instead of doing polynomial multiplication $Y(Z)=F(Z)\,X(Z)$ as digital
convolution, we can perform the following operations:

1. Apply the digital Fourier transform to vectors $\mathbf{f}$ and $\mathbf{x}$ to create vectors $\mathbf{F}$ and $\mathbf{X}$.
2. Perform point-by-pont multiplication $Y_k=F_k\,X_k$ for $k=0,1,\cdots,N-1$.
3. Apply the inverse Fourier transform to go from $\mathbf{Y}$ to $\mathbf{y}$.
"""

# ╔═╡ c69e864f-d7e5-47db-9654-b73a7b3e4cec
md"""
!!! note

    Convolution in the time domain corresponds to multiplication in the Fourier domain.
"""

# ╔═╡ cc437932-3057-4060-8141-8a2511c0307e
md"""
The adjoint of convolution corresponds to multiplication by $F(1/Z)$ or the complex-conjugate $\bar{F}_k$. Correspondingly, the auto-correlation $A(Z)=F(Z)\,F(1/Z)$ corresponds to the *spectrum* $A_k=F_k\,\bar{F}_k=|F_k|^2$.
"""

# ╔═╡ e917dff6-f28a-45cc-a0a2-cbd171171715
md"""
![](https://fftw.org/fftw-logo-med.gif)

[https://fftw.org/](https://fftw.org/)
"""

# ╔═╡ 15b51e23-edba-4a2a-b9cd-fadd11791159
function convolve(x::Array, b::Array, adjoint=false)
    nx, nb = length(x), length(b)
    y = zeros(eltype(x), nx)
    for ib in 1:nb, iy in ib:nx
        if (adjoint) # correlation
            y[iy + 1 - ib] += x[iy] * b[ib]
        else         # convolution
            y[iy] += x[iy + 1 - ib] * b[ib]
        end
    end
    return y
end

# ╔═╡ ad37f4f3-11d1-4d35-9b27-6ae0d7e1d5b9
begin
	inp = zeros(Float32,30)
	for (k,a) in Dict(6=>1,15=>0.5,16=>1,17=>0.5,25=>-1)
	    inp[k] = a
	end
	filt = map(x -> 1-x/4, 0:3);
end

# ╔═╡ 57f848d6-c761-41a7-979f-2098a8335d24
conv = convolve(inp, filt, false);

# ╔═╡ 587a7ab8-971d-4958-9a2c-91d106db6e5d
corr = convolve(inp, filt, true);

# ╔═╡ a1019abd-7860-4a99-a5db-7fb3f1ca1347
function stems(data, label, color) 
    plt=plot(zeros(Float32, 30), label=:none, color=:black)
    plot!(plt, data, line=:stem, marker=:circle, 
          label=label, color=color, legend=:outerleft, 
          xlim=[0, 32.5], ylim=[-1.7, 1.7], border=:none)   
    return plt
end

# ╔═╡ cc757ba2-43aa-4679-b38b-1c4bc4a2f5f6
plot(stems(inp, "input", :blue), 
     stems(filt, "filter", :green), 
     stems(conv, "convolution", :red),
     stems(corr, "correlation", :red),
     layout=(4, 1))

# ╔═╡ cdf245f3-fc40-4f34-8e77-845a9d19ab0d
Finput = rfft(inp);

# ╔═╡ 3ccca937-ab13-4f97-a4f8-69d5479c89ce
filtpad = vcat(filt, zeros(26));

# ╔═╡ 9fb1fad6-b785-448e-91b7-4b4c22a34250
Ffilt = rfft(filtpad);

# ╔═╡ 56591c80-b98f-4f0c-89b4-15b12f2b6582
fconv = irfft(Finput .* Ffilt, 30);

# ╔═╡ 143152ef-3c41-4fd5-bab2-88d30fbb191e
plot(stems(inp, "input", :blue), 
     stems(abs.(Finput)/2, "Fourier[input]", :purple),
     stems(abs.(Ffilt)/2, "Fourier[filter]", :purple),
     stems(fconv, "convolution", :red),
     layout=(4, 1))

# ╔═╡ 54020d5f-6737-499a-a35c-52f4293e9f54
function fconvolve(x::Array{T}, b::Array{T}, adjoint=false) where T <: Real
    "Convolution by Fourier transform"
    nx, nb = length(x), length(b)
    # pad with zeros
    bpad = vcat(b, zeros(nx-nb))
    Fx, Fb = rfft(x), rfft(bpad)
    if (adjoint) 
        return irfft(Fx .* conj(Fb), nx)
    else
        return irfft(Fx .* Fb, nx)
    end
end

# ╔═╡ 9d0791e2-c9f5-4d49-b282-94aeb15fd7d5
md"""
!!! assignment
    ## Task 1

    Is it computationally beneficial to perform convolution using Fourier transform? Let's find out.

    Generate a random data array of size 1024 and a random filter of size 8. Use `Benchmark Tools` to compare the computational efficiency of convolution via Fourier transform (`fconvolve()`) with that of convolution in the time domain (`convolve()`). By changing the filter length, find out experimentally at which length the Fourier-based convolution becomes competitive.
"""

# ╔═╡ 11137186-041a-44c9-9c50-d578285361d8
md"""
### Derivative property

Taking the derivative of $f(t)$ and using the definition of the inverse Fourier transform produces

$$\frac{d f}{d t} = i\omega\,\frac{1}{2\pi}\,\int F(\omega)\,e^{i\omega\,t}\,dt=i\omega\,F(\omega)\;.$$

This equation shows that the derivative operator in the time domain corresponds to multiplication by $i\omega$ in the Frequency domain. 
"""

# ╔═╡ 52cdf9bc-3233-45d2-a27c-9cb188c686c0
md"""
Similarly, the second derivative corresponds to multiplication by $(i\omega)^2=-\omega^2$, and the $n$-th derivative corresponds to multiplication by $(i\omega)^n$. This fact has immediate applications in using the Fourier transform to solve differential equations.
"""

# ╔═╡ 7468276e-a30e-4aff-a742-1fabc7161664
md"""
The derivative property also gives us an intuitive understanding of what low and high frequencies mean. The Fourier transform of a smooth function with infinitely many continuous derivatives will exhibit exponential decay at high frequencies (faster than any power of $|\omega|$). If a function is continuous with $n-1$ derivatives but has a discontinuity in the $n$-th derivative, its Fourier transform will decay at high frequencies as $1/|\omega|^{n+1}$. Therefore, we can use low- and high-frequency separation to decompose the data into smooth and rough components.
"""

# ╔═╡ ff4e8531-0e44-4f7e-9a6f-c10f64c92788
md"""
## FFT algorithm

Direct multiplication by the dense Fourier matrix would require $O(N^2)$ operations, which is prohibitively expensive for large-scale data. Fortunately, the matrix's structure enables a more efficient algorithm.
"""

# ╔═╡ 5f941792-b282-4346-a2c4-9ac522967238
md"""
Let us take a closer look at the discrete Fourier transform and suppose that we want to compute the Fourier
transform of size $2\,N$ 

$$F_{k}^{[2N]}[\mathbf{f}] = \sum\limits_{n=0}^{2N-1} f_n\,e^{-i\,2\,\pi\,k\,n/(2N)}\;.$$
"""

# ╔═╡ b2a460e3-1cf7-499e-92c3-d921a21faa9a
md"""
Dividing the sum into two parts to
collect the even and odd elements of the vector $\mathbf{f}$, we can notice that

$$\begin{array}{rcl}F_{k}^{[2N]}[\mathbf{f}] & = & \displaystyle \sum\limits_{n=0}^{N-1} f_{2n}\,e^{-i\,2\,\pi\,k\,2n/(2N)} \\
& + & \displaystyle \sum\limits_{n=0}^{N-1} f_{2n+1}\,e^{-i\,2\,\pi\,k\,(2n+1)/(2N)} \end{array}$$
$$= \sum\limits_{n=0}^{N-1} f_{2n}\,e^{-i\,2\,\pi\,k\,n/N} + e^{-i\,2\,\pi\,k/(2N)}\,\sum\limits_{n=0}^{N-1} f_{2n+1}\,e^{-i\,2\,\pi\,k\,n/N}\;.$$
"""

# ╔═╡ 6b148257-0170-4be6-8af9-7c49895f38a1
md"""
Both sums in the right-hand side of the equation are just Fourier transforms of the size $N$:

$$F_{k}^{[2N]}[\mathbf{f}] = F_{k}^{[N]}[\mathbf{f}_{\mbox{even}}]+e^{-i\,\frac{\pi\,k}{N}}\,F_{k}^{[N]}[\mathbf{f}_{\mbox{odd}}]\;.$$
"""

# ╔═╡ e0f76f74-41d1-4b2b-b3bb-e5dab61655fa
md"""
We see that by dividing the input into even and odd parts, we can split the work and reduce the cost of computing the Fourier transform of size $2N$ to two Fourier transforms of size $N$ plus $O(N)$ operations for the multiplication by $e^{-i\,\frac{\pi\,k}{N}}$.

This splitting is the familiar *divide and conquer* strategy we previously saw in the quicksort algorithm. As in the case of quicksort, the cost equation

$$C(N) = 2\,C(N/2)+O(N)$$

with $C(1)=0$ resolves to $C(N)=O(N\,\log N)$. 
"""

# ╔═╡ 86d1a6e4-38ac-4ec7-a677-4dc89db9ab65
md"""
As with quicksort, the reduction in cost from $O(N^2)$ to $O(N\,\log N)$ turns an algorithm from unaffordably slow to comfortably fast. This approach is the FFT (*Fast Fourier Transform*) algorithm.
"""

# ╔═╡ bd3d9adb-01c9-4878-abd9-e596d6bb8722
function slowft(f::Vector{T}) where T <: Complex
    "slow digital Fourier transform"
    N = length(f)
    F = zeros(T,N)
    # straightforward matrix multiplication
    for k in 1:N, n in 1:N
        F[k] += f[n] * exp(-im*2π*(k-1)*(n-1)/N)
    end
    return F
end

# ╔═╡ 347117b6-1829-4142-9ed0-f5422e55ad8b
data = rand(ComplexF64,64);

# ╔═╡ 56fe02e9-8c86-4000-b382-0933c43fc8c6
@assert(fft(data) ≈ slowft(data))

# ╔═╡ eadb2be1-971e-4475-8011-64b34aa03a93
md"""
!!! assignment
    ## Task 2

    We can attempt to accelerate the slow Fourier transform algorithm by recognizing that the matrix elements repeat and precomputing them.
"""

# ╔═╡ ea47966c-f279-46b2-94a9-0fe60901ed06
function fasterft(f::Vector{T}) where T <: Complex
    "faster digital Fourier transform"
    N = length(f)
    # precompute elements of the Fourier matrix
    M = similar(f)
    Zn, Z = one(T), exp(-im*2π/N)
    for n in 1:N
        M[n] = Zn
        Zn *= Z
    end
    F = zeros(T, N)
    # matrix multiplication
    for k in 1:N, n in 1:N
        # % outputs the division remainder
        F[k] += f[n] * M[(k-1)*(n-1) % N + 1]
    end
    return F
end

# ╔═╡ bef9014a-e476-461c-b608-46df48202bac
@assert(fft(data) ≈ fasterft(data))

# ╔═╡ 2dd44ccc-87a0-4e7a-a7b5-0a1b255f820e
md"""
**Your task**: use `BenchmarkTools` to compare the efficiency of `slowft()` and `fasterft()`.
"""

# ╔═╡ 6510e777-4657-4361-8484-030a16ecb360
function fastft(f::Vector{T}) where T <: Complex
    "fast digital Fourier transform"
    N = length(f) # assume N is a power of 2
    if N <= 1; return f; end
    eve = fastft(f[1:2:N-1]) # 1,3,5,...
    odd = fastft(f[2:2:N])   # 2,4,6,...
    N ÷= 2 # divide by two
    F = similar(f)
    for k in 1:N
        a = exp(-im*π*(k-1)/N)
        F[k]   = eve[k] + a*odd[k]
        F[k+N] = eve[k] - a*odd[k]
    end
    return F
end

# ╔═╡ 9e38d00f-66f1-44f2-be91-f9dd3f5e0942
@assert(fft(data) ≈ fastft(data))

# ╔═╡ 951beb31-84c9-414b-b7c6-85c2424e6b63
np = pyimport("numpy");

# ╔═╡ c9c85020-b8b4-487c-aa18-988d5ee39616
@assert(fft(data) ≈ np.fft.fft(data))

# ╔═╡ bac4eb3a-616d-4879-84f6-e6c24141df5f
Conda.add("scipy")

# ╔═╡ a01a5dc5-32a1-4f78-a3b5-9848bddd68f3
scipy = pyimport("scipy");

# ╔═╡ 9ff746c2-efa6-4e9d-85c7-322947ab034d
@assert(fft(data) ≈ scipy.fft.fft(data))

# ╔═╡ 7b53016e-9186-41b2-ac2c-d40267c17c33
# dictionary of functions
fourier = Dict(
    "Slow (own version)" => slowft,
    "Fast (own version)" => fastft,
    "Julia FFTW" => fft,
    "Python NumPy library" => np.fft.fft,
    "Python SciPy library" => scipy.fft.fft
    )

# ╔═╡ 6e513422-7341-4997-8507-4e86645f746a
begin
	df = DataFrame(Implementation = String[], Time = Float64[])
	r = rand(ComplexF64, 2^12)
	for (name, func) in fourier
	    # @belapsed extracts the runtime in seconds
	    time = @belapsed $func($r);
	    push!(df, [name, time])
	end
	sort!(df, :Time)
end

# ╔═╡ 4ad850be-e6d5-4cdc-adc2-bacc811c551e
md"""
!!! note
   
    The FFT algorithm uses the divide-and-conquer strategy to reduce the computational cost of the discrete Fourier transform from $O(N^2)$ to $O(N\,\log N)$.
"""

# ╔═╡ 1dc186c5-b773-42a2-97d5-2a0b2175b431
md"""
### References

The FFT algorithm is due to Cooley and Tukey (1965). Some evidence of earlier work was reported in the literature, by Vern Herbert at Chevron in the early 1960s and even by Carl Friedrich Gauss in 1805 (Heideman et al., 1984). 

**Cooley, J. W., and J. W. Tukey, 1965, An algorithm for the machine calculation of complex 
Fourier series: Mathematics of Computation, 19, 297–301.**

**Heideman, M. T., D. H. Johnson, and C. S. Burrus, 1984, Gauss and the history of the fast 
Fourier transform: IEEE ASSP Magazine, 1, 14–21.**
"""

# ╔═╡ 1db510a3-5d9a-4e96-8de5-807207cdee9e
md"""
Van Loan (1991) describes different versions of the FFT algorithm. Frigo & Jonhnson (2005) describe the FFTW implementation.

**Van Loan, C., 1991, Computational frameworks for the fast Fourier transform: SIAM.**

**Frigo, M., and S. G. Johnson, 2005, The design and implementation of FFTW3: Proceedings 
of the IEEE, 93, 216–231.**
"""

# ╔═╡ 855bbbfe-17af-4d9f-84c2-54b18cb207bc
md"""
!!! assignment
    ## Task 3

    We can compare the efficiency of various algorithms systematically using different data sizes.
"""

# ╔═╡ a48f1de4-4031-47fe-9c4a-1ff0300e2169
begin
	nmin, nmax = 4, 10
	na = nmax - nmin + 1
	sizes = [2^n for n in nmin:nmax]
	arrays = map(n -> rand(ComplexF32, n), sizes)
end

# ╔═╡ d168207a-e015-4ff8-a5d5-583a081e7559
begin
	# create a data frame
	bm = DataFrame(Sizes = sizes)
	algorithms= Dict("slowft" => slowft, "fastft" => fastft)
	for (name, func) in algorithms
	    times = Array{Float64}(undef,na)
	    for n in 1:na
	        a = arrays[n]
	        times[n] = @belapsed $func($a);
	    end
	    bm[!,name] = times
	end
end

# ╔═╡ 8fe8e34f-76e2-47df-98f5-4a086f89701a
bm

# ╔═╡ 6c05593c-f0f5-45b5-83c8-eec8c2c9ce1b
plot(sizes, [bm.slowft bm.fastft], labels=["slowft" "fastft"], 
    linewidth=3, markershape=:diamond, xscale=:log2,
	yscale=:log10, legend=:top, 
	xlabel="data size", ylabel="computing time (seconds)",
    title="Comparison of Fourier Transform Algorithms")

# ╔═╡ 49cf0a78-e0d4-4cc1-b6d8-f9a004d91fd0
md"""
**Your task**: add `fasterft` and some other implementation of the FFT algorithm to the plot.
"""

# ╔═╡ 2441a7ab-30f6-47af-ac9f-f0824e4be40e
md"""
## Multidimensional Fourier transform

The multidimensional Fourier transform is defined as a chain of 1-D
transforms applied in different directions. The cost of transforming
multidimensional data of size $N=N_{1}\,N_{2}\,\cdots\,N_{d}$ is

$$\begin{array}{c}\frac{N}{N_1}\,O(N_1\,\log N_1) + \cdots
+ \frac{N}{N_d}\,O(N_d\,\log N_d) \\ = O(d\,N\,\log N)\;.\end{array}$$
"""

# ╔═╡ 5bf2361e-e190-49fe-8892-417831e08425
download("https://ahay.org/data/viking/viking.rsf@", "stack.bin")

# ╔═╡ 18a7ff97-a1bb-4260-88e4-a0e01fed301b
begin
	nt, nx = 1001, 2142
	# single-precision array
	stack = Array{Float32}(undef, nt, nx) 
	read!("stack.bin", stack)
end

# ╔═╡ 44bd1e40-9b3d-4652-a0fb-cad88fa4df97
begin
	dt=0.004 
	dx=0.0125  
	t=range(start=0, step=dt, length=nt)
	x=range(start=0, step=dx, length=nx)
end

# ╔═╡ db6a9170-72de-4189-b4bd-5ed920f14fa8
function plot2(image, title, x, xlabel, y, ylabel)
    array = vec(image)
    clip = quantile(abs.(array), 0.9)
    return heatmap(x, y, image, yflip=:true, color=:grays, 
                   legend=:none, clim=(-clip,clip),
                   xlabel=xlabel, ylabel=ylabel, title=title)
end

# ╔═╡ 2eeaef90-6079-40c6-b168-8571b471f9eb
ptx = plot2(stack,L"$p(t,x)$", x, L"$x$ [km]", t, L"$t$ [s]")

# ╔═╡ 99f35f71-27bb-41be-a14f-606464dc056d
md"""
* **Keys, R. G., and D. J. Foster, 1998, Comparison of seismic inversion methods on a single real data set: Society of Exploration Geophysicists.**
"""

# ╔═╡ bf1caba1-8c5a-4ccd-88d2-24adde8c0b04
begin
	ft = plan_rfft(t);
	fstack = mapslices(trace -> ft * trace, stack; dims=1);
	f = rfftfreq(nt, 1/dt);
end

# ╔═╡ da517690-a3d1-40cf-b7ab-3cfdf33658b2
begin
	fx = plan_fft(x);
	kstack = mapslices(slice -> fftshift(fx * slice), stack; dims=2);
	k = fftshift(fftfreq(nx, 1/dx));
end

# ╔═╡ 73cac177-f99b-4e78-8bd7-4e6f75c63af4
fkstack = mapslices(slice -> fftshift(fx * slice), 
	                fstack; dims=2);

# ╔═╡ 943435c9-b346-4ea3-a7c9-6287609fe51d
begin
	pfx = plot2(real(fstack), L"$P(\omega,x)$", 
	            x, L"$x$ [km]", f, L"$\omega$ [Hz]");
	ptk = plot2(real(kstack), L"$P(t,k)$", k, 
	            L"$k$ [1/km]", t, L"$t$ [s]");
	pfk = plot2(real(fkstack), L"$P(\omega,k)$", k, 
	            L"$k$ [1/km]", f, L"$\omega$ [Hz]");
	plot(ptx, ptk, pfx, pfk, layout=(2, 2))
end

# ╔═╡ 4bc06054-f7f8-422d-abbf-1e791e3c08f6
test = rand(ComplexF64,2^10);

# ╔═╡ 42a4932f-f300-4629-b09a-314b8d8109d0
pf = plan_fft(test);

# ╔═╡ f79bf3f0-c11d-46f1-83f8-c54b428094f4
@btime fft(test);

# ╔═╡ ae242b59-d01f-455e-844b-1953207fa511
@btime pf * test;

# ╔═╡ f21fd04c-8507-42aa-8367-9eab2076665c
md"""
## Cosine Fourier transform

When dealing with real-valued signals, it is sometimes helpful to use a real-to-real version of the Fourier transform, specifically the cosine transform.

In the case of continuous signals, the cosine transform has the form

$$C(\omega) = \int\limits_{0}^{\infty} f(t)\,\cos(\omega\,t)\,dt$$
"""

# ╔═╡ 520cf4a7-e042-4af2-b610-773ce2499c95
md"""
with the inverse transform given by

$$f(t) = \frac{2}{\pi}\,\int\limits_{0}^{\infty} C(\omega)\,\cos(\omega\,t)\,d\omega\;.$$

To derive this relationship from the Fourier transform, we can assume that $f(t)$ is defined only for positive time $t$ and extend to negative time as an even function $f(-t) = f(t)$. In this case, $F(\omega)$ becomes a real-valued even function $F(\omega)=2\,C(\omega)$.
"""

# ╔═╡ da2baed4-4976-4c42-837e-e05cb180fea6
md"""
In the case of discrete signals, the digital cosine transform can be computed analogously with the help of the digital Fourier transform and is usually defined as

$$C_k = \frac{f_0}{2} + \sum\limits_{n=1}^{N-1} f_n\,\cos\left(\pi\,\frac{k\,n}{N}\right) + (-1)^k\,\frac{f_N}{2}$$

for $k=0,1,\cdots,N$ and $n=0,1,\cdots,N$ with the inverse transform

$$f_n = \frac{2}{N}\,\left[\frac{C_0}{2} + \sum\limits_{k=1}^{N-1} C_k\,\cos\left(\pi\,\frac{k\,n}{N}\right) + (-1)^n\,\frac{C_N}{2}\right]\;.$$
"""

# ╔═╡ e2b2d90d-bbaa-4b6a-8e57-fbab0ac16f7b
function cosft(f::Vector{T}) where T <: Real
    N = length(f) # assume N is a power of 2 + 1
    N2 = 2*(N-1)
    f2 = Vector{T}(undef,N2)
    # extend symmetrically 
    f2[1:N] = f
    f2[N+1:N2] = f[N-1:-1:2]
    ft = rfft(f2)
    return real(ft)
end

# ╔═╡ 1f935738-dc0d-449c-8bae-f2116fc55d7e
function icosft(f::Vector{T}) where T <: Real
    N = length(f) # assume N is a power of 2 + 1
    N2 = 2*(N-1)
    ft = irfft(complex(f),N2)
    return ft[1:N]
end

# ╔═╡ 6f4c5d8a-32f5-4d84-ac51-644bd8e415d7
ctest = rand(Float32, 2^12+1);

# ╔═╡ 4f3855d8-78d6-4874-90f9-a6ec1bfc229f
@assert(ctest ≈ icosft(cosft(ctest)))

# ╔═╡ fb046441-b91d-4978-a6e4-9d8adbe145c3
md"""
## Kolmogorov spectral factorization

We met before the Wilson-Burg method of spectral factorization. An alternative approach involves the use of the Fourier transform. It follows the observation that applying the exponential function preserves the causality of a time series. If $U(Z)=u_0+u_1\,Z+u_2\,Z^2+\cdots$ is causal, so is $e^{U(Z)}=1+U(Z)+U(Z)^2/2+\cdots$.
"""

# ╔═╡ 9af4ef22-4979-45b7-a3c1-fa2d3e459d6e
md"""
To find a minimum-phase filter $S(Z)$ from its autocorrelation $A(Z)=S(Z)\,S(1/Z)$, the method of spectral factorization attributed to Andrey Kolmogorov uses the Fourier transform in the following steps:

1. Apply the Fourier transform to the autocorrelation. With some abuse of notation, we will denote it $A(\omega)=|S(\omega)|^2$. 
2. Apply the logarithm function

$$W(\omega) = \log[A(\omega)]\;.$$
"""

# ╔═╡ 7b58b904-92b7-45f3-9a94-9aff4f9f44a1
md"""
3. Apply the inverse Fourier transform.
4. Separate the causal part of the corresponding time series. The series $W(Z)$ will be symmetric because its Fourier transform is real.

$$W(Z) = U(Z)+U(1/Z)\;.$$

5. Apply the Fourier transform to go back to the frequency domain.
6. Take the exponent

$$S(\omega) = \exp[U(\omega)]\;.$$
"""

# ╔═╡ d27c9dd9-5350-4492-91a9-a7b6d420d36d
md"""
7. Apply the inverse Fourier transform to reconstruct the coefficients of $S(Z)$.

Thus, the method works at the cost of four fast Fourier transforms.
"""

# ╔═╡ f563dba5-72b2-4cd8-8c4f-c3a803a1758c
md"""
* **Kolmogorov, A. N., 1939, Sur l'interpolation et extrapolation des suites stationnaires: CR Acad. Sci. Paris, 208, 2043-2045.**
* **Claerbout, J. F., 1976, Fundamentals of geophysical data processing: Blackwell.**
* **Sayed, A. H., and T. Kailath, 2001, A survey of spectral factorization methods: Numer. Linear Algebra Appl., 8, 467-496.**
"""

# ╔═╡ a8eed79d-ca4b-4f7d-9550-909c65cc213b
function kolmogorov(f::Vector{T}, pad=16) where T <: Real
    "convert a filter to equivalent minimum-phase"
    nt = length(f)  
    nf = pad*nextprod((2, 3, 5), nt)         # optimal length 
    fourier = rfft(vcat(f, zeros(T, nf-nt))) # F(Z) -> F(ω)
    auto = abs.(fourier) .^ 2                # A(ω) = |F(ω)|^2
    ϵ = eps(T)^2                             # to avoid log(0)
    auto = log.(auto .+ ϵ)                   # W(ω) = log[A(ω)]
    back = irfft(complex(auto), nf)          # W(ω) -> W(Z)
    back[1] = zero(T)                        # W(Z) -> U(Z)
    back[nf÷2+1:nf] .= zero(T)
    fourier = rfft(back)                     # U(Z) -> U(ω)
    fourier = exp.(fourier)                  # S(ω) = exp[U(ω)]
    back = irfft(fourier, nf)                # S(ω) -> S(Z)
    return back[1:nt]
end

# ╔═╡ a2d1b81d-9106-41a8-bd20-02c647c728d5
F = Polynomial(filt,:Z)

# ╔═╡ 20455025-79ad-49a2-bc7c-5a19757fa3ac
R = roots(F)

# ╔═╡ 686a0f0e-58d6-4601-a8a4-524f0bbd65a5
begin
	R2 = R[:]
	R2[2]=1/R[2]
	R2[3]=1/R[3]
	F2 = fromroots(R2, var=:Z)
	filt2 = coeffs(F2)
end

# ╔═╡ 1cab903d-1804-4633-9f5e-2d1ab5c350c7
@assert filt ≈ kolmogorov(filt2)

# ╔═╡ 623ba534-51a5-4b5c-b503-ed30a9f9fa21
md"""
!!! assignment
    ## Task 4

    Try a different example filter from the previous computational assignment to test the Kolmogorov algorithm.
"""

# ╔═╡ 50cf83d8-381f-4160-a38a-609aa2759440
import Random

# ╔═╡ 72dbe22c-210f-4910-9428-e3103adc2b4b
begin
	Random.seed!(2025) # for reproducibility
	c = randn(ComplexF64, 10)
	c = vcat(c, conj(c)) # assure real coefficients
	P = fromroots(c, var=:Z)
	p = coeffs(P)
end

# ╔═╡ e22b8302-b374-412d-906c-f9cf30ab40ac
stems(p/100,"mixed-phase", :red)

# ╔═╡ eb222f8b-ab9c-4431-8e32-5a6c52139b81
md"""
Apply the Kolmogorov algorithm to the mixed-phase filter and verify that the result is a minimum-phase filter.
"""

# ╔═╡ 146ac798-1e6c-4854-8fc3-25e8675f8c89
md"""
## Data example

This exercise explores how the compactness of the Fourier transform can be used to reconstruct missing data.
"""

# ╔═╡ f343a868-c1f1-424c-ad17-632ae3ca3072
download("https://ahay.org/data/hall/horizon.asc","horizon.asc")

# ╔═╡ 4ac6f3d1-5cae-493d-a410-12e042fac84a
xyz = readdlm("horizon.asc") # read data from a text file

# ╔═╡ a2595346-af2b-4b2d-bd0e-12f8fa6790b8
begin
	n1, n2 = 196, 291
	iline = xyz[1:n1:end, 1]
	xline = xyz[1:n1, 2]
end

# ╔═╡ f2cc14a8-53ec-4ca0-970b-091d410310db
# taper edges to ensure periodicity
function taper!(x, nw)
    n1, n2 = size(x)
    for i in 1:nw
        w = sin(0.5*π*(i-1)/nw)
        w *= w
        for i1 in 1:n1
            x[i1,i] *= w
            x[i1,n2+1-i] *= w
        end
        for i2 in 1:n2
            x[i,i2] *= w
            x[n1+1-i,i2] *= w
        end
    end
end

# ╔═╡ ed6b00d4-6d51-45ce-b95b-6ebc0ccfe379
begin
	horizon=reshape(xyz[:, 3], (n1, n2))
	# subtract mean
	mean = sum(horizon)/length(horizon)
	horizon .-= mean
	taper!(horizon, 25)
end

# ╔═╡ b06afebe-56b9-4b5f-b25d-4bfb8be3f117
plot_horizon(map, title) = heatmap(iline, xline, map, title=title, 
    xlabel=L"$x$ (m)", ylabel=L"$y$ (m)", clim=(-14,14))

# ╔═╡ 879279a8-c1a5-481e-b477-540d062e8161
plot_horizon(horizon, "Seismic Horizon")

# ╔═╡ 367d0ad7-0f27-4201-a05a-3be2d0d43739
begin
	# 2-D complex-valued FFT
	fhorizon = fftshift(fft(complex(horizon)));
	k1 = fftshift(fftfreq(n1, 0.1))
	k2 = fftshift(fftfreq(n2, 0.1))
end

# ╔═╡ 63efb91d-7074-4ab3-9e40-bf90e0dcd547
plot_fourier(map, title) = heatmap(k2, k1, abs.(map), title=title, 
    xlabel=L"$k_x$ (1/m)", ylabel=L"$k_y$ (1/m)", clim=(0,1000))

# ╔═╡ 2bf474c4-cd9c-413b-8df5-813384dc6bc1
plot_fourier(fhorizon, "Fourier Transform")

# ╔═╡ efb5958e-8656-44d8-a14e-626e3274a891
md"""
We will create missing parts by artificially cutting holes in the original data.
"""

# ╔═╡ abcb414a-4bc6-48c2-97a4-99d7fbe44467
begin
	# cut holes
	input = deepcopy(horizon)
	input[126:147, 151:172] .= 0
	input[151:172, 51:72] .= 0
	input[76:97, 126:147] .= 0
end

# ╔═╡ b5f730e7-9051-4009-8e56-1cdeb79d23ba
plot_horizon(input, "Input")

# ╔═╡ 29edc257-affc-4e4c-a221-2a49db8350f5
finput = fftshift(fft(complex(input)));

# ╔═╡ 1ec42728-0895-4720-b462-8532eb0abf8c
plot_fourier(finput, "Fourier Transform")

# ╔═╡ ff723ef3-51ef-4efc-85c0-80753c6bf4bd
md"""
We observe that the original data's support in the Fourier domain is limited by its smoothness. Introducing holes in the physical domain creates discontinuities that extend the Fourier response beyond the original support.
"""

# ╔═╡ 142e1970-7d02-4221-9b52-acea2e48f5a2
md"""
### Projection onto convex sets

To accomplish the task of missing data interpolation, we will use an iterative method known as POCS (*projection onto convex
sets*). 

By definition, a convex set $\mathcal{C}$ is a set of functions such that, for any $f_1(\mathbf{x})$ and $f_2(\mathbf{x})$
from the set, $g(\mathbf{x}) = \lambda\,f_1(\mathbf{x}) + (1-\lambda)\,f_2(\mathbf{x})$ (for $0 \le \lambda \le 1$) also belongs
to the set. A projection onto a convex set means finding a function in the set that is of the shortest distance to the given function. 
"""

# ╔═╡ 0bd21ab0-9a25-438b-ac0e-9a0aba267b5f
md"""
The POCS theorem states that a function that belongs to the intersection of two convex sets $C_1$ and $C_2$ can be found iteratively by alternating projections onto the two sets.

**Youla, D.C. and Webb, H., 1982. Image Restoration by the Method of Convex Projections: Part 1: Theory. IEEE Transactions on Medical Imaging, 1, 81-94.**
"""

# ╔═╡ cf50003d-feb6-490d-85a9-567cc7f1bbd5
md"""
![](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Projections_onto_convex_sets_circles.svg/567px-Projections_onto_convex_sets_circles.svg.png)

[https://en.wikipedia.org/wiki/Projections\_onto\_convex\_sets](https://en.wikipedia.org/wiki/Projections_onto_convex_sets)
"""

# ╔═╡ db8bd0cf-45a0-4dad-ac6c-e145addeccca
md"""
 In our example, $C_1$ is the set of all functions equal to the known data outside the holes. $C_2$ is the set of all functions with predefined compact support in the Fourier domain (and therefore are smooth in the physical domain). The algorithm consists of the following steps:

1. Apply 2-D Fourier transform. 
2. Apply a mask to enforce compact support.
3. Apply inverse 2-D Fourier transform.
4. Replace data outside of the holes with known data.
5. Repeat.
"""

# ╔═╡ 8564550d-fb07-41bd-bd95-349846c4e3f8
function pocs(image, xmask, fmask, niter)
    "Projection onto convex sets"
    filled = deepcopy(image)
    for iter in 1:niter
        ft = fftshift(fft(complex(filled))) # forward FT
        ft[fmask] .= zero(eltype(ft))       # Fourier mask
        filled = real(ifft(ifftshift(ft)))  # inverse FT
        filled[xmask] = image[xmask]        # space mask
    end
    return filled
end

# ╔═╡ 59a4bde1-81d4-44fb-9d21-51694a9283b8
begin
	xmask = ones(Bool, size(horizon))
	xmask[126:147, 151:172] .= false
	xmask[151:172, 51:72] .= false
	xmask[76:97, 126:147] .= false
end

# ╔═╡ 4c099ade-3273-4b1b-b274-779937bda0b2
plot_horizon(xmask*14, "Space Mask")

# ╔═╡ a697fd38-a0fa-453c-93be-250f7f5bc120
begin
	fmask = ones(Bool, size(finput))
	for i1 in 1:n1, i2 in 1:n2
	    x, y = k1[i1], k2[i2]
	    if x*x + y*y < 0.00005
	        fmask[i1,i2] = false
	    end
	end
end

# ╔═╡ c1dc117c-fa62-4378-ab2b-784f06c77f3b
plot_fourier(fmask*1000, "Fourier Mask")

# ╔═╡ c0bd03b1-af5c-4773-a6b9-727ef4946a0c
niter = 5 # number of iterations

# ╔═╡ 0d0e5551-784c-49f2-9a4f-2c64b84496ea
filled = pocs(input, xmask, fmask, niter);

# ╔═╡ 2cbb1e0e-7e7e-4e65-b9c3-6a83e0200f90
plot_horizon(filled, "Filled After $niter Iterations")

# ╔═╡ 54ed1a81-a16e-4b62-b277-8f223cd6f634
md"""
!!! assignment
    ## Task 5

    Experimentally determine the number of POCS iterations needed to reach convergence and create an animated GIF showing the output throughout the iterations.
"""

# ╔═╡ f35b139c-858c-4f49-870f-df9539a29110
md"""
!!! assignment
    ## Bonus Task

    Improve the POCS algorithm's convergence by creating a more effective Fourier mask. You can also switch the program from complex-valued `fft` to real-to-complex `rfft` to boost efficiency.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Conda = "8f4d0f93-b110-5947-807f-2305c1781a2d"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUIExtra = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"
Polynomials = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
BenchmarkTools = "~1.5.0"
Conda = "~1.10.2"
DataFrames = "~1.8.2"
FFTW = "~1.10.0"
HTTP = "~1.11.0"
LaTeXStrings = "~1.4.0"
Plots = "~1.41.6"
PlutoUIExtra = "~0.1.8"
Polynomials = "~4.1.1"
PyCall = "~1.96.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "ff15cf0b807d151e23c5416b46babc418dc3fafa"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "2eeb2c9bef11013efc6f8f97f32ee59b146b09fb"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.44"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "21d088c496ea22914fe80906eb5bce65755e5ec8"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.1"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "b19db3927f0db4151cb86d073689f2428e524576"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataPipes]]
git-tree-sha1 = "3fb39158bc35c984cac5edb1ff55daa88a4b5074"
uuid = "02685ad9-2d12-40c3-9f73-c6aeda6a7ff5"
version = "0.3.19"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e86f4a2805f7f19bec5129bc9150c38208e5dc23"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.4"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8f05e9a2e7c2e3eb524102bb2926c5743c07fbe1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "cac41ca6b2d399adfc95e51240566f8a60a80806"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6866aec60ef98e3164cd8d6855225684207e9dff"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.12+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.FlexiMaps]]
deps = ["Accessors", "DataPipes", "InverseFunctions"]
git-tree-sha1 = "c2e79264c5e749d099d7ae854f64ec73f2f9e3e9"
uuid = "6394faf6-06db-4fa8-b750-35ccc60383f7"
version = "0.1.29"

    [deps.FlexiMaps.extensions]
    AxisKeysExt = "AxisKeys"
    DictionariesExt = "Dictionaries"
    IntervalSetsExt = "IntervalSets"
    StructArraysExt = "StructArrays"
    UnitfulExt = "Unitful"

    [deps.FlexiMaps.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dictionaries = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "9e0fb9e54594c47f278d75063980e43066e26e20"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "44716a1a667cb867ee0e9ec8edc31c3e4aa5afdc"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.24"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "be8a1b8065959e24fdc1b51402f39f3b6f0f6653"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.24+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IntervalSets]]
git-tree-sha1 = "79d6bd28c8d9bccc2229784f1bd637689b256377"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.14"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "5d5e0a78e971354b1c7bff0655d11fdc1b0e12c8"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.4"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "cb20a4eacda080e517e4deb9cfb6c7c518131265"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.6"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "0ecd70a51c13e150266e76a865f10a64a7f178a3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.82"

[[deps.PlutoUIExtra]]
deps = ["AbstractPlutoDingetjes", "ConstructionBase", "FlexiMaps", "HypertextLiteral", "InteractiveUtils", "IntervalSets", "Markdown", "PlutoUI", "Random", "Reexport"]
git-tree-sha1 = "b4ff5d24e2dc8fbf319cd44f9f81b5356e27bafb"
uuid = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"
version = "0.1.8"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "Setfield", "SparseArrays"]
git-tree-sha1 = "2d99b4c8a7845ab1342921733fa29366dae28b24"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.1"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"
    PolynomialsRecipesBaseExt = "RecipesBase"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "624de6279ab7d94fc9f672f0068107eb6619732c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.3.2"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "d5b7dd0e226774cbd87e2790e34def09245c7eab"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "ed349d26affcacafbc7fc2941ace1fb98f71e715"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+1"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "da8c1f6eee04831f14edcfa5dae611d309807e57"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.3.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╟─37dbf60a-86ab-4bd3-bf70-4362a1054df6
# ╠═a295efe1-1cce-4fd6-a6d4-039765cd86de
# ╠═f2492ddd-d71a-4d1a-932c-a314de792f93
# ╟─0ddbb258-0d65-4541-8ea7-37f4220bc934
# ╟─64618f52-c56d-4cc5-89f4-6f8b5348b290
# ╟─2f2ec598-9e50-4f25-8179-b118c39e783c
# ╟─ef185e26-899c-4ebc-bf82-153c1e5d6223
# ╟─047cd9b1-2800-49a3-9350-ae003f37902c
# ╟─47d4f2bd-1580-40e1-bac2-5f527a2ef7fc
# ╠═d287df06-9393-4817-b0a2-b687ec3d1997
# ╠═2119a94f-f033-4e94-96de-d5caa4048cef
# ╠═9b700931-4446-4758-b211-745bb2dc3429
# ╠═c14c9130-1186-48df-b7c6-6424000c17ba
# ╟─da8af9ef-7baf-4602-9dcc-e87c091b10a7
# ╟─73fed2f1-5075-43f4-86f5-f6bc7ad778a8
# ╟─be83061d-c0a5-4091-bd20-3b62d3f8510f
# ╟─d8c93879-0785-42cc-b110-f32b8db6964c
# ╟─e83191d8-9bfc-4222-a86b-65e07667d448
# ╟─eb8e5c8d-cfda-4010-b666-7ec41ee06902
# ╟─b549c4a2-edc3-4837-972b-d56c003ed6ae
# ╟─7e1baaef-e73a-49e2-baed-e1e620ff0712
# ╟─c69e864f-d7e5-47db-9654-b73a7b3e4cec
# ╟─cc437932-3057-4060-8141-8a2511c0307e
# ╟─e917dff6-f28a-45cc-a0a2-cbd171171715
# ╠═b697c05d-5bea-44ec-a88b-bc6df24fd7fa
# ╠═15b51e23-edba-4a2a-b9cd-fadd11791159
# ╠═ad37f4f3-11d1-4d35-9b27-6ae0d7e1d5b9
# ╠═57f848d6-c761-41a7-979f-2098a8335d24
# ╠═587a7ab8-971d-4958-9a2c-91d106db6e5d
# ╠═a1019abd-7860-4a99-a5db-7fb3f1ca1347
# ╠═cc757ba2-43aa-4679-b38b-1c4bc4a2f5f6
# ╠═cdf245f3-fc40-4f34-8e77-845a9d19ab0d
# ╠═3ccca937-ab13-4f97-a4f8-69d5479c89ce
# ╠═9fb1fad6-b785-448e-91b7-4b4c22a34250
# ╠═56591c80-b98f-4f0c-89b4-15b12f2b6582
# ╠═143152ef-3c41-4fd5-bab2-88d30fbb191e
# ╠═54020d5f-6737-499a-a35c-52f4293e9f54
# ╟─9d0791e2-c9f5-4d49-b282-94aeb15fd7d5
# ╟─11137186-041a-44c9-9c50-d578285361d8
# ╟─52cdf9bc-3233-45d2-a27c-9cb188c686c0
# ╟─7468276e-a30e-4aff-a742-1fabc7161664
# ╟─ff4e8531-0e44-4f7e-9a6f-c10f64c92788
# ╟─5f941792-b282-4346-a2c4-9ac522967238
# ╟─b2a460e3-1cf7-499e-92c3-d921a21faa9a
# ╟─6b148257-0170-4be6-8af9-7c49895f38a1
# ╟─e0f76f74-41d1-4b2b-b3bb-e5dab61655fa
# ╟─86d1a6e4-38ac-4ec7-a677-4dc89db9ab65
# ╠═bd3d9adb-01c9-4878-abd9-e596d6bb8722
# ╠═347117b6-1829-4142-9ed0-f5422e55ad8b
# ╠═56fe02e9-8c86-4000-b382-0933c43fc8c6
# ╟─eadb2be1-971e-4475-8011-64b34aa03a93
# ╠═ea47966c-f279-46b2-94a9-0fe60901ed06
# ╠═bef9014a-e476-461c-b608-46df48202bac
# ╟─2dd44ccc-87a0-4e7a-a7b5-0a1b255f820e
# ╠═6510e777-4657-4361-8484-030a16ecb360
# ╠═9e38d00f-66f1-44f2-be91-f9dd3f5e0942
# ╠═889e94b1-7e20-4ba0-85bf-31023d0090d3
# ╠═951beb31-84c9-414b-b7c6-85c2424e6b63
# ╠═c9c85020-b8b4-487c-aa18-988d5ee39616
# ╠═de7ddbcb-4417-46a3-ba8c-a0f9f30f1915
# ╠═bac4eb3a-616d-4879-84f6-e6c24141df5f
# ╠═a01a5dc5-32a1-4f78-a3b5-9848bddd68f3
# ╠═9ff746c2-efa6-4e9d-85c7-322947ab034d
# ╠═7b53016e-9186-41b2-ac2c-d40267c17c33
# ╠═17b5cd2e-284a-4dd3-bc28-5d2f815ff3b7
# ╠═6e513422-7341-4997-8507-4e86645f746a
# ╟─4ad850be-e6d5-4cdc-adc2-bacc811c551e
# ╟─1dc186c5-b773-42a2-97d5-2a0b2175b431
# ╟─1db510a3-5d9a-4e96-8de5-807207cdee9e
# ╟─855bbbfe-17af-4d9f-84c2-54b18cb207bc
# ╠═a48f1de4-4031-47fe-9c4a-1ff0300e2169
# ╠═d168207a-e015-4ff8-a5d5-583a081e7559
# ╠═8fe8e34f-76e2-47df-98f5-4a086f89701a
# ╠═6c05593c-f0f5-45b5-83c8-eec8c2c9ce1b
# ╟─49cf0a78-e0d4-4cc1-b6d8-f9a004d91fd0
# ╟─2441a7ab-30f6-47af-ac9f-f0824e4be40e
# ╠═5bf2361e-e190-49fe-8892-417831e08425
# ╠═18a7ff97-a1bb-4260-88e4-a0e01fed301b
# ╠═44bd1e40-9b3d-4652-a0fb-cad88fa4df97
# ╠═42bf53fb-6164-4187-ae78-a6a6c859a1b7
# ╠═db6a9170-72de-4189-b4bd-5ed920f14fa8
# ╠═2eeaef90-6079-40c6-b168-8571b471f9eb
# ╟─99f35f71-27bb-41be-a14f-606464dc056d
# ╠═bf1caba1-8c5a-4ccd-88d2-24adde8c0b04
# ╠═da517690-a3d1-40cf-b7ab-3cfdf33658b2
# ╠═73cac177-f99b-4e78-8bd7-4e6f75c63af4
# ╠═943435c9-b346-4ea3-a7c9-6287609fe51d
# ╠═4bc06054-f7f8-422d-abbf-1e791e3c08f6
# ╠═42a4932f-f300-4629-b09a-314b8d8109d0
# ╠═f79bf3f0-c11d-46f1-83f8-c54b428094f4
# ╠═ae242b59-d01f-455e-844b-1953207fa511
# ╟─f21fd04c-8507-42aa-8367-9eab2076665c
# ╟─520cf4a7-e042-4af2-b610-773ce2499c95
# ╟─da2baed4-4976-4c42-837e-e05cb180fea6
# ╠═e2b2d90d-bbaa-4b6a-8e57-fbab0ac16f7b
# ╠═1f935738-dc0d-449c-8bae-f2116fc55d7e
# ╠═6f4c5d8a-32f5-4d84-ac51-644bd8e415d7
# ╠═4f3855d8-78d6-4874-90f9-a6ec1bfc229f
# ╟─fb046441-b91d-4978-a6e4-9d8adbe145c3
# ╟─9af4ef22-4979-45b7-a3c1-fa2d3e459d6e
# ╟─7b58b904-92b7-45f3-9a94-9aff4f9f44a1
# ╟─d27c9dd9-5350-4492-91a9-a7b6d420d36d
# ╟─f563dba5-72b2-4cd8-8c4f-c3a803a1758c
# ╠═a8eed79d-ca4b-4f7d-9550-909c65cc213b
# ╠═ed8e7bfa-ccc9-423f-838a-94ae025326ce
# ╠═a2d1b81d-9106-41a8-bd20-02c647c728d5
# ╠═20455025-79ad-49a2-bc7c-5a19757fa3ac
# ╠═686a0f0e-58d6-4601-a8a4-524f0bbd65a5
# ╠═1cab903d-1804-4633-9f5e-2d1ab5c350c7
# ╟─623ba534-51a5-4b5c-b503-ed30a9f9fa21
# ╠═50cf83d8-381f-4160-a38a-609aa2759440
# ╠═72dbe22c-210f-4910-9428-e3103adc2b4b
# ╠═e22b8302-b374-412d-906c-f9cf30ab40ac
# ╟─eb222f8b-ab9c-4431-8e32-5a6c52139b81
# ╟─146ac798-1e6c-4854-8fc3-25e8675f8c89
# ╠═d1097cf8-e5e5-4902-b3d4-387ad103fbb0
# ╠═f343a868-c1f1-424c-ad17-632ae3ca3072
# ╠═4ac6f3d1-5cae-493d-a410-12e042fac84a
# ╠═a2595346-af2b-4b2d-bd0e-12f8fa6790b8
# ╠═f2cc14a8-53ec-4ca0-970b-091d410310db
# ╠═ed6b00d4-6d51-45ce-b95b-6ebc0ccfe379
# ╠═b06afebe-56b9-4b5f-b25d-4bfb8be3f117
# ╠═879279a8-c1a5-481e-b477-540d062e8161
# ╠═367d0ad7-0f27-4201-a05a-3be2d0d43739
# ╠═63efb91d-7074-4ab3-9e40-bf90e0dcd547
# ╠═2bf474c4-cd9c-413b-8df5-813384dc6bc1
# ╟─efb5958e-8656-44d8-a14e-626e3274a891
# ╠═abcb414a-4bc6-48c2-97a4-99d7fbe44467
# ╠═b5f730e7-9051-4009-8e56-1cdeb79d23ba
# ╠═29edc257-affc-4e4c-a221-2a49db8350f5
# ╠═1ec42728-0895-4720-b462-8532eb0abf8c
# ╟─ff723ef3-51ef-4efc-85c0-80753c6bf4bd
# ╟─142e1970-7d02-4221-9b52-acea2e48f5a2
# ╟─0bd21ab0-9a25-438b-ac0e-9a0aba267b5f
# ╟─cf50003d-feb6-490d-85a9-567cc7f1bbd5
# ╟─db8bd0cf-45a0-4dad-ac6c-e145addeccca
# ╠═8564550d-fb07-41bd-bd95-349846c4e3f8
# ╠═59a4bde1-81d4-44fb-9d21-51694a9283b8
# ╠═4c099ade-3273-4b1b-b274-779937bda0b2
# ╠═a697fd38-a0fa-453c-93be-250f7f5bc120
# ╠═c1dc117c-fa62-4378-ab2b-784f06c77f3b
# ╠═c0bd03b1-af5c-4773-a6b9-727ef4946a0c
# ╠═0d0e5551-784c-49f2-9a4f-2c64b84496ea
# ╠═2cbb1e0e-7e7e-4e65-b9c3-6a83e0200f90
# ╟─54ed1a81-a16e-4b62-b277-8f223cd6f634
# ╟─f35b139c-858c-4f49-870f-df9539a29110
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
