### A Pluto.jl notebook ###
# v0.20.27

#> [frontmatter]
#> chapter = 4
#> section = 7
#> order = 7
#> image = "https://raw.githubusercontent.com/fonsp/Pluto.jl/580ab811f13d565cc81ebfa70ed36c84b125f55d/demo/plutodemo.gif"
#> title = "7 Data Analysis Notebook"
#> tags = ["module4", "track_julia", "track_material", "Pluto", "PlutoUI"]
#> layout = "layout.jlhtml"

using Markdown
using InteractiveUtils

# ╔═╡ 07e27cd5-4c74-4e73-b1a7-d5fe2651f02f
begin
	using PlutoUIExtra
	TableOfContents()
end

# ╔═╡ 868cfea3-4029-416a-95e3-81f4e05a7a81
using Plots

# ╔═╡ 58b6375b-9067-4ee2-b260-4b0ae3b8e426
using LinearAlgebra, Random

# ╔═╡ 0a8c1424-35f5-44f7-97bb-8ba02df829ec
import HTTP

# ╔═╡ 0b62a901-60f5-4eec-a5e3-e92f880ff6c6
let
    notebook_dir = @__DIR__
    toc_path = joinpath(notebook_dir, "TOC_Notebook.jl")
    
    Markdown.parse("[⬅️ Back to Table of Contents](./open?path=$(HTTP.escapeuri(toc_path)))")
end

# ╔═╡ c8439cda-03b0-44a0-b5ad-14617f30ba32
md"""
# Helix transform


Extending convolution (polynomial multiplication in the $Z$-transform notation) to the multidimensional case is straightforward. Less obvious is how to extend inverse convolution (polynomial division). We will discuss how to accomplish this using Claerbout's helix transform.
"""

# ╔═╡ 7beae512-736a-48ff-af57-41e7f805f599
md"""
## Multidimensional convolution

Extending convolution to multiple dimensions is straightforward. Instead of shifting in a single dimension (represented by $Z$ in the $Z$-transform notation), we must consider shifts across multiple dimensions and their combinations. Therefore, convolution with a two-dimensional filter can be expressed as

$$y_{k_1,k_2} = \sum\limits_{n_2=0}^{N_2-1} \sum\limits_{n_2=0}^{N_2-1} f_{n_1,n_2}\,x_{k_1-n_1,k_2-n_2}$$

or as a multiplication by a polynomial filter

$$F_2(Z_1,Z_2)=f_{00}+f_{10}\,Z_1+f_{01}\,Z_2+f_{20}\,Z_1^2+f_{11}\,Z_1\,Z_2+f_{02}\,Z_2^2+\cdots\;,$$

where $Z_1$ and $Z_2$ represent shifts in two orthogonal directions. Analogously, convolution with an $N$-dimensional filter can be represented as a multiplication by

$$\begin{array}{rcl}F_N(Z_1,Z_2,\cdots,Z_N)=f_{00\cdots0} & + & f_{100\cdots0}\,Z_1+f_{010\cdots0}\,Z_2+\cdots+f_{00\cdots01}\,Z_N \\
                                  & + & f_{200\cdots0}\,Z_1^2+f_{110\cdots0}\,Z_1\,Z_2+\cdots\end{array}$$
"""

# ╔═╡ 233d6871-5184-487a-b631-9b6de417eabd
function convolve(x, b, adjoint=false)
    "N-dimensional convolution"
    nx, nb = map(CartesianIndices, (x,b))
    y = zeros(eltype(x), size(x))
    for ib in nb, iy in ib:last(nx)
        ix = iy - ib + first(nx)
        if adjoint # correlation
            y[ix] += x[iy] * b[ib]
        else       # convolution
            y[iy] += x[ix] * b[ib]
        end
    end
    return y
end

# ╔═╡ 908b43bb-6980-42ef-8b58-d97e299ff66d
begin
	z = zeros(10,5)
	for ci in CartesianIndices(z)
		println(ci)
	end
end

# ╔═╡ ba18eae6-c56c-4838-85d5-2a37bdaba6c1
begin
	spike = zeros(Float32, 17, 13)
	spike[8,2] = 1
	horse = zeros(Float32, 3, 4)
	horse[3,:] .= 1
	horse[:,1] .= 1
	slant = zeros(Float32, 4, 8)
	slant[1,1] = 1
	slant[2,6] = -0.4
	slant[3,7] = 0.3
	slant[4,8] = -0.9;
end

# ╔═╡ 4d409430-0f82-44df-ab59-1a9108f1e6d4
begin
	ihorse = convolve(spike, horse);
	islant = convolve(spike, slant);
	conv2 = convolve(ihorse, slant);
end

# ╔═╡ e0781ad3-5c46-4cb6-baac-5ab20a02dbfa
plot2(data, title) = heatmap(data, color=:grays, legend=:none, 
	                  aspect_ratio=:equal, title=title,
                      yflip=:true, xlim=(0.5,size(data,2)+0.5), 
	                  ylim=(0.5,size(data,1)+0.5), 
                      clim=(-1,1), axis=nothing)

# ╔═╡ c87909ed-82da-4c0b-9cd7-41b91166c693
begin
	p1 = plot2(ihorse, "A");
	p2 = plot2(islant, "B");
	p3 = plot2(conv2, "A (convolve) B");
	plot(p1, p2, p3, layout=(1, 3))
end

# ╔═╡ 0a686f45-2554-4e2b-9b5f-27212ecc044c
md"""
A more challenging task is inverse convolution (polynomial division). The key to recursive division in one dimension was creating an ordered input sequence to generate the output recursively. This ordering is not immediately obvious in the multidimensional case. The helix transform establishes it.
"""

# ╔═╡ 5327b862-bd75-4fad-93d0-c6898ea9c673
md"""
## Helix transform

To make a concrete example, consider a 2-D filter $F_2(Z_1,Z_2)=1+2\,Z_1+3\,Z_1\,Z_2$. The convolution of this filter with a $5 \times 3$ input can be expressed as a multiplication by a
$15 \times 15$ matrix

$$\left[
\begin{array}{l}
y_{00} \\
y_{10} \\
y_{20} \\
y_{30} \\
y_{40} \\
\hline
y_{01} \\
y_{11} \\
y_{21} \\
y_{31} \\
y_{41} \\
\hline
y_{02} \\
y_{12} \\
y_{22} \\
y_{32} \\
y_{42} 
\end{array}
\right] = 
\left[
\begin{array}{rrrrr|rrrrr|rrrrr}
  1 & 0 & 0 & 0 & 0
&   &   &   &   & 
&   &   &   &   &    \\
  2 & 1 & 0 & 0 & 0 
&   &   &   &   & 
&   &   &   &   &    \\
  0 & 2 & 1 & 0 & 0  
&   &   & 0 &   & 
&   &   & 0 &   &    \\
  0 & 0 & 2 & 1 & 0 
&   &   &   &   & 
&   &   &   &   &    \\
  0 & 0 & 0 & 2 & 1
&   &   &   &   & 
&   &   &   &   &    \\
\hline
  0 & 0 & 0 & 0 & 0 
& 1 & 0 & 0 & 0 & 0 
&   &   &   &   &    \\
  3 & 0 & 0 & 0 & 0
& 2 & 1 & 0 & 0 & 0
&   &   &   &   &    \\
  0 & 3 & 0 & 0 & 0
& 0 & 2 & 1 & 0 & 0  
&   &   & 0 &   &    \\
  0 & 0 & 3 & 0 & 0 
& 0 & 0 & 2 & 1 & 0 
&   &   &   &   &    \\
  0 & 0 & 0 & 3 & 0 
& 0 & 0 & 0 & 2 & 1 
&   &   &   &   &    \\
\hline
  0 & 0 & 0 & 0 & 0
& 0 & 0 & 0 & 0 & 0 
& 1 & 0 & 0 & 0 & 0   \\
  0 & 0 & 0 & 0 & 0
& 3 & 0 & 0 & 0 & 0
& 2 & 1 & 0 & 0 & 0   \\
  0 & 0 & 0 & 0 & 0
& 0 & 3 & 0 & 0 & 0
& 0 & 2 & 1 & 0 & 0   \\
  0 & 0 & 0 & 0 & 0
& 0 & 0 & 3 & 0 & 0
& 0 & 0 & 2 & 1 & 0   \\
  0 & 0 & 0 & 0 & 0
& 0 & 0 & 0 & 3 & 0
& 0 & 0 & 0 & 2 & 1 
\end{array}
\right]\,
\left[
\begin{array}{l}
x_{00} \\
x_{10} \\
x_{20} \\
x_{30} \\
x_{40} \\
\hline
x_{01} \\
x_{11} \\
x_{21} \\
x_{31} \\
x_{41} \\
\hline
x_{02} \\
x_{12} \\
x_{22} \\
x_{32} \\
x_{42} 
\end{array}
\right]\;.$$
    
We observe that the 2-D convolution matrix has a block-diagonal structure similar to that of a 1-D convolution matrix. However, some of the diagonals appear broken. The idea behind the helix transform is to keep matrix values constant along all diagonals. In other words, we replace the previous matrix with the new matrix

$$\left[
\begin{array}{rrrrr|rrrrr|rrrrr}
  1 & 0 & 0 & 0 & 0
&   &   &   &   & 
&   &   &   &   &    \\
  2 & 1 & 0 & 0 & 0 
&   &   &   &   & 
&   &   &   &   &    \\
  0 & 2 & 1 & 0 & 0  
&   &   & 0 &   & 
&   &   & 0 &   &    \\
  0 & 0 & 2 & 1 & 0 
&   &   &   &   & 
&   &   &   &   &    \\
  0 & 0 & 0 & 2 & 1
&   &   &   &   & 
&   &   &   &   &    \\
\hline
  0 & 0 & 0 & 0 & \mathbf{2} 
& 1 & 0 & 0 & 0 & 0 
&   &   &   &   &    \\
  3 & 0 & 0 & 0 & 0
& 2 & 1 & 0 & 0 & 0
&   &   &   &   &    \\
  0 & 3 & 0 & 0 & 0
& 0 & 2 & 1 & 0 & 0  
&   &   & 0 &   &    \\
  0 & 0 & 3 & 0 & 0 
& 0 & 0 & 2 & 1 & 0 
&   &   &   &   &    \\
  0 & 0 & 0 & 3 & 0 
& 0 & 0 & 0 & 2 & 1 
&   &   &   &   &    \\
\hline
  0 & 0 & 0 & 0 & \mathbf{3}
& 0 & 0 & 0 & 0 & \mathbf{2} 
& 1 & 0 & 0 & 0 & 0   \\
  0 & 0 & 0 & 0 & 0
& 3 & 0 & 0 & 0 & 0
& 2 & 1 & 0 & 0 & 0   \\
  0 & 0 & 0 & 0 & 0
& 0 & 3 & 0 & 0 & 0
& 0 & 2 & 1 & 0 & 0   \\
  0 & 0 & 0 & 0 & 0
& 0 & 0 & 3 & 0 & 0
& 0 & 0 & 2 & 1 & 0   \\
  0 & 0 & 0 & 0 & 0
& 0 & 0 & 0 & 3 & 0
& 0 & 0 & 0 & 2 & 1 
\end{array}
\right]\;.$$

This transformation involves specific boundary conditions in which the 2-D filter wraps around the boundary with a one-sample shift. This shift makes the 2-D convolution operator equivalent to 1-D convolution with a filter that has gaps.
"""

# ╔═╡ 00e9dff1-4d9c-4d66-ad0e-d33525646c9e
md"""
![](https://ahay.org/RSF/book/gee/hlx/Fig/sergey-helix.png)

* **Claerbout, J., 1998, Multidimensional recursive filters via a helix: Geophysics, 63, 1532-1541.**
* **Fomel, S., and J. Claerbout, 2003, Multidimensional recursive filter preconditioning in geophysical estimation problems: Geophysics, 68, 409-420.**
* **Mersereau, R. M., and D. E. Dudgeon, 1974, The representation of two-dimensional sequences as one-dimensional sequences: IEEE Trans. on Acoustics, Speech, and Signal Processing, SSP-22, 320-325.**
"""

# ╔═╡ a0d7a53c-a85c-4d94-bcef-925f1ada4138
md"""
![](https://m.media-amazon.com/images/I/71oGlhJPpVL._UF1000,1000_QL80_.jpg)
"""

# ╔═╡ 5ccde062-0cf9-4fcd-944f-850e80a81e46
md"""
The formal definition of the helix transform is the mapping 

$$\begin{array}{rcl}Z_1 & = & Z\;, \\
Z_2 & = & Z^{N_1}\;, \\
\cdots & & \cdots \\
Z_k & = & Z^{N_1\,N_2\,\cdots\,N_k}\;.\end{array}$$

Thus, the 2-D filter $F_2(Z_1,Z_2)=1+2\,Z_1+3\,Z_1\,Z_2$ maps to the 1-D helical filter $F_1(Z)=1+2\,Z+3\,Z^6$ when applied on a $5 \times 3$ grid or the 1-D helical filter $F_1(Z)=1+2\,Z+3\,Z^{501}$ when applied on a $500 \times 300$ grid.
"""

# ╔═╡ b308544d-342f-4aaa-9eb4-9e9d0e877645
md"""
!!! note
  
    The helix transform imposes specific boundary conditions that make multidimensional convolution equivalent to one-dimensional convolution with a filter that has gaps.
"""

# ╔═╡ edc19bb9-8929-4117-b040-8cb9cee02a25
md"""
!!! assignment

    ## Task 1 (theoretical)

    Find a three-dimensional filter $F_3(Z_1,Z_2,Z_3)$ that corresponds to the helical filter $H(Z) = 1 + 2 Z + 3\,Z^{101} + 4\,Z^{701}$ on a $25 \times 25 \times 25$ grid.
"""

# ╔═╡ 157f5f20-58e3-41eb-97b9-f9ff590153b8
md"""
To implement helical filtering, we first create a data structure that allows us to skip over gapped values in the filter and perform calculations using only non-zero filter values.
"""

# ╔═╡ 0659951e-47c5-4115-b3c0-48ef4d5e1c15
mutable struct HelixFilter
    lag::Vector{CartesianIndex}
    flt::Vector
    HelixFilter(lag,flt) = new(map(CartesianIndex,lag),flt)
end

# ╔═╡ 2fc1789d-bbf6-4689-8040-68277c5b5dc3
Base.length(a::HelixFilter) = Base.length(a.lag)

# ╔═╡ 2cd510b3-7d52-4fc2-b54e-8a584fb501f8
function helix(a::HelixFilter, ci::CartesianIndices)
    "convert helix lags to 1-D for a given grid"
    # middle of the grid
    mid = CartesianIndex(Tuple(last(ci)) .÷ 2)
	na = length(a.lag)
	lag = Vector{Int}(undef, na)
	lin = LinearIndices(ci)
	for i in 1:na
        lag[i] = lin[a.lag[i] + mid] - lin[mid]
    end
    return lag
end

# ╔═╡ a6277f35-419b-4c08-97bd-e68c668a9b5d
function hconvolve(x, a::HelixFilter, adjoint=false)
    cx, nx, na = CartesianIndices(x), length(x), length(a)
    lag = helix(a, cx)
    y = deepcopy(x) # accounting for the zero coefficient
    for ia in 1:na
        for iy in 1 + lag[ia] : nx
            ix = iy - lag[ia]
            if adjoint # correlation
                y[ix] += x[iy] * a.flt[ia]
            else       # convolution
                y[iy] += x[ix] * a.flt[ia]
            end
        end
    end
    return y
end

# ╔═╡ f73c8d74-8c19-47b0-8548-aa5fc5fd8cdc
md"""
Regarding computational cost, the helix transform offers no advantage over standard multidimensional convolution. Its computational advantage lies in enabling inverse convolution or polynomial division.
"""

# ╔═╡ 105c82f6-ba75-4118-b4d2-6804796af16f
function hrecursive(x, a::HelixFilter, adjoint=false)
    cx, nx, na = CartesianIndices(x), length(x), length(a)
    lag = helix(a, cx)
    y = similar(x)
    if adjoint
        @inbounds for ix in nx:-1:1
            t = x[ix]
            for ia in 1:na
                iy = ix + lag[ia]
                if iy <= nx
                    t -= a.flt[ia] * y[iy]
                end
            end
            y[ix] = t
        end
    else
        @inbounds for ix in 1:nx
            t = x[ix]
            for ia in 1:na
                iy = ix - lag[ia]
                if iy >= 1
                    t -= a.flt[ia] * y[iy]
                end
            end
            y[ix] = t
        end
    end
    return y
end

# ╔═╡ 9cad3b34-e5fb-4512-aa33-57341c14d06a
horse

# ╔═╡ 96319f90-4af2-45d2-bc70-68e7b09da964
h_horse = HelixFilter([(1,0),(2,0),(2,1),(2,2),(2,3)], ones(Float32, 5))

# ╔═╡ 0346e4ce-7150-4afa-b506-747ac6047d31
slant

# ╔═╡ cc9fd071-8c16-4e17-9614-a43045abb148
h_slant = HelixFilter([(1,5),(2,6),(3,7)], [-0.4, 0.3, -0.9])

# ╔═╡ f75dad75-f590-414b-8c9b-25050f225cd4
begin
	hhorse = hconvolve(spike, h_horse);
	hslant = hconvolve(spike, h_slant);
	hconv2 = hconvolve(hhorse, h_slant);
end

# ╔═╡ cf0d841f-1dbb-4276-aad9-e8b3def57392
begin
	hp1 = plot2(hhorse, "A");
	hp2 = plot2(hslant, "B");
	hp3 = plot2(hconv2, "A (convolve) B");
	plot(hp1, hp2, hp3, layout=(1, 3))
end

# ╔═╡ c4d612dc-73fe-41ec-92e1-666fcc8f2662
begin
	spike2 = zeros(Float32, 30,20)
	spike2[8,12] = -1
	spike2[7,3] = 1
end

# ╔═╡ dc5e6482-1fb9-4830-8f30-9626118f02e6
begin
	spike3 = zeros(Float32, 30,20)
	spike3[24,18] = -1
	spike3[15,6] = 1
end

# ╔═╡ 330404b7-6695-4585-bcf9-e9933900279f
four = HelixFilter([(1,0),(-1,1),(0,1),(1,1)],-ones(Float32,4)/4.0)

# ╔═╡ 87bd88cd-d8a3-4d0f-927f-77f77c669fb1
finput = hconvolve(spike2,four) + spike3;

# ╔═╡ 19b733ed-1d7f-4525-82de-7ac152e60190
divis = hrecursive(finput,four);

# ╔═╡ 132b5edd-52e8-47a5-a68e-5f4448b9533d
begin
	fp1 = plot2(finput, "input");
	fp2 = plot2(divis, "input/filter");
	plot(fp1, fp2, layout=(1, 2))
end

# ╔═╡ 108e4423-f61f-4bbf-b053-fbea44eab5a7
md"""
!!! assignment

    ## Task 2
    Let us check whether the helical forward and adjoint convolutions pass the dot-product test.
"""

# ╔═╡ 13e36512-7ced-4a91-9e35-32b34ffd8cfa
function dottest(forward::Function, adjoint::Function, 
                 m::Array, d::Array)
    "Dot-product test"
    mod = similar(m); rand!(mod)
    dat = similar(d); rand!(dat)
    println(" L[m]⋅d = $(forward(mod) ⋅ dat)")
    println("L'[d]⋅m = $(adjoint(dat) ⋅ mod)")
end

# ╔═╡ cc8a8ca1-2f66-4574-afbd-ab2649f3329f
dottest(x -> hconvolve(x, h_horse, false), 
        x -> hconvolve(x, h_horse, true),
        spike, spike)

# ╔═╡ 137d4887-fb9a-4c72-8318-b98c05143b41
md"""
**Your task**: Perform the dot-product test for the recursive convolution in `hrecursive`.
"""

# ╔═╡ 59f2dc5b-d9bb-41e6-b166-e33e976696d4
md"""
## Spectral factorization on a helix

The Wilson-Burg algorithm can be adapted for use with helical filters.
"""

# ╔═╡ 801118cb-70da-4ffc-aca2-c41931337966
function hwilson(au0::Real, auto::HelixFilter, lag, niter=10, pad=5)
    "Wilson-Burg spectral factorization on a helix"
    # initialize filter
    na = length(lag)
    T = eltype(auto.flt)
    a = HelixFilter(lag,zeros(T, na))
    grid = Tuple(pad*(maximum(a.lag)-minimum(a.lag)))
    au = zeros(T, grid)
    na2 = length(auto)
    nc = CartesianIndex(grid .÷ 2)
    # initialize autocorrelation
    au[nc] = au0
    for i in 1:na2
        au[nc+auto.lag[i]] = auto.flt[i]
        au[nc-auto.lag[i]] = auto.flt[i]
    end
    b = zeros(T,grid)
    b[nc] = one(T)
    ic = LinearIndices(au)[nc]
    for iter in 1:niter
        bb = hrecursive(au, a, false)  # S/A
        cc = hrecursive(bb, a, true)   # S/(AA')
        ϵ = zero(T)
        for i in ic+1:2*ic-1 # b = Causal[1+cc]
            bi = 0.5*(cc[i] + cc[2*ic-i]) / cc[ic] 
            if ϵ < abs(bi); ϵ = abs(bi); end
            b[i] = bi
        end
        @show iter, ϵ
        b = hconvolve(b, a) # c = A b
        for i in 1:na
            a.flt[i] = b[nc+a.lag[i]]
        end
    end
    return a
end

# ╔═╡ 278f7dea-9887-4dfc-af36-cea4ebc51039
begin
	spike1 = zeros(Float32, (20, 30))
	spike1[10,15] = 1
end

# ╔═╡ d2bef630-c655-4b96-8445-37e8e3c5e1db
same = hconvolve(spike1, h_horse, false);

# ╔═╡ 310173c7-f937-4411-b38d-578172009ec8
auto = hconvolve(same, h_horse, true);

# ╔═╡ eb6f694f-77d8-42bb-87c5-fa8c4eb0b865
function autocorr(a::HelixFilter)
    "computing autocorrelation"
    na, T = length(a), eltype(a.flt)
    counts = Dict{CartesianIndex,T}()
    for i in 1:na
        counts[a.lag[i]] = get(counts, a.lag[i], zero(T)) + a.flt[i]
        for j in i+1:na
            lag = a.lag[j] - a.lag[i]
            counts[lag] = get(counts, lag, zero(T)) + a.flt[j]*a.flt[i]
        end
    end
    n = length(counts)
    lag = Vector{CartesianIndex}(undef,n)
    flt = Vector{T}(undef,n)
    idx = 1
    for (k,v) in counts
        lag[idx] = k
        flt[idx] = v
        idx += 1
    end
    return HelixFilter(lag, flt)
end

# ╔═╡ 6a43dd30-7799-4509-a175-0459770f7488
a_horse = autocorr(h_horse)

# ╔═╡ 4ae0dd57-182c-47f0-8243-f0ed9f3c7ff9
begin
	a0 = 1 + sum(h_horse.flt .^ 2)
	lag = vcat(
	    [(x,0) for x in 1:5],
	    [(x,1) for x in -5:5],
	    [(x,2) for x in -5:5],
	    [(x,3) for x in -5:2])
	f_horse = hwilson(a0, a_horse, lag)
end

# ╔═╡ fb604622-7044-4564-8773-059d08e06715
filt = hconvolve(spike1, f_horse, false);

# ╔═╡ 970126c6-8e4a-4b01-9f1a-80290e372f85
lspike2 = hrecursive(hrecursive(auto, f_horse, false), f_horse, true);

# ╔═╡ fb5730c4-c250-4979-af8e-20c337a6393b
begin
	ap1 = plot2(same, "input");
	ap2 = plot2(auto/maximum(auto), "autocorr");
	ap3 = plot2(filt, "Wilson-Burg factor");
	ap4 = plot2(lspike2/maximum(lspike2), "autocorr/(factor*factor')");
	plot(ap1, ap2, ap3, ap4, layout=(2, 2))
end

# ╔═╡ a058667e-f3a8-47e5-83b1-73a8bf7680c4
md"""
## Helical derivative

*Helical derivative* is a filter obtained by spectral factorization of the Laplacian filter

$$L(Z_1,Z_2) = 4 - Z_1 - 1/Z_1 - Z_2 - 1/Z_2\;.$$

To invert the Laplacian filter, we use a helix, where it takes the form

$$L_H(Z) = 4 - Z - Z^{-1} - Z^{N_1} - Z^{-N_1}\;,$$

and factor it into two minimum-phase parts $L_H(Z) = D(Z)\,D(1/Z)$ using the Wilson-Burg algorithm. The helical derivative $D(Z)$ enhances the image but is not confined to a single direction.
"""

# ╔═╡ 37282a3e-faf6-4dc1-be12-30dbb2e2189c
# download a data file
download("https://ahay.org/data/bay/mount.rsf@","data.bin")

# ╔═╡ 538a1c40-047f-4624-9957-d7b3b2335005
mount = Array{Float32}(undef, 979, 1400); # single-precision array

# ╔═╡ 15f4a40c-366d-4e44-b490-33694f5749fa
read!("data.bin", mount)

# ╔═╡ 764abb6c-da30-4fe6-bba3-28e4c91a7b6f
heatmap(mount, color=:grays,
        title="Digital Elevation Map of Mount St. Helens")

# ╔═╡ ca01a561-a004-4168-b4c0-414d74529b1f
function deriv(x::Vector{T}) where T <: Real
    # defivative filter D(Z) = Z-1
    n = length(x)
    y = similar(x)
    for i in 1:n-1
        y[i] =  x[i+1] - x[i]
    end
    y[n] = zero(T)
    return y
end

# ╔═╡ d52c06c2-2be5-489e-8582-3ef27d156c99
d1 = mapslices(deriv, mount; dims=1);

# ╔═╡ 16c3eca9-03f3-4d01-abdf-92c1b969da0d
heatmap(d1, color=:grays, clim=(-30,30), title="Vertical Derivative")

# ╔═╡ 20a68a1e-02ce-4ba8-a227-eae93233fc74
begin
	laplacian = HelixFilter([(1,0),(0,1)],[-1.0,-1.0])
	hlag = vcat([(x,0) for x in 1:10],[(x,1) for x in -10:0])
	helder = hwilson(4.0, laplacian, hlag)
	# make sure the filter removes DC
    helder.flt /= sum(-helder.flt)
end

# ╔═╡ 7367d669-656c-4333-bfc9-1222a89d93f1
begin
	lspike = zeros(Float32,25,25)
	lspike[13,13] = 1
	
	lap = hconvolve(lspike,laplacian,false) + hconvolve(lspike,laplacian,true)
	lap[13,13] = 4
	lp1 = plot2(lap/4,"Laplacian");
	
	frw = hrecursive(lap,helder,false)
	lp2 = plot2(frw/3.2,"Laplacian/Helder");
	
	adj = hrecursive(lap,helder,true)
	lp3 = plot2(adj/3.2,"Laplacian/Helder'");
	
	inv = hrecursive(frw,helder,true)
	lp4 = plot2(inv,"Laplacian/Helder/Helder'");
	
	plot(lp1, lp2, lp3, lp4, layout=(2, 2))
end

# ╔═╡ 89eb1ba3-20c0-4c26-b8b6-47ab0c15c3d6
dh = hconvolve(mount, helder, false);

# ╔═╡ ef0c5550-69e4-4848-806a-1394a096d57d
heatmap(dh, color=:grays, clim=(-20,20), title="Helical Derivative")

# ╔═╡ fcc8b334-b706-4c52-b4bc-a01cbfd09ab1
savefig("heldir.png")

# ╔═╡ bc4cc41a-1208-4c7f-8cad-2131a0e192e5
md"""
!!! assignment

    ## Task 3

	We used a five-point Laplacian filter to define the helical derivative.

    |     |     |    |
    |-----|-----|----|
	|  0  | -1  |  0 | 
    | -1  |  4  | -1 |
    |  0  | -1  |  0 |

    A more accurate version is the nine-point filter

    |     |     |    |
    |-----|-----|----|
	| -1  | -4  | -1 | 
    | -4  | 20  | -4 |
    | -1  | -4  | -1 |

    $\begin{array}{rcl}\hat{L}_9(Z_1,Z_2) = 20 & - & 4\,Z_1 - 4\,Z_1^{-1} - 4\,Z_2 - 4\,Z_2^{-1} \\ & - & Z_1\,Z_2 - Z_1\,Z_2^{-1} - Z_2\,Z_1^{-1} - Z_1^{-1}\,Z_2^{-1}\;.\end{array}$

    Implement and test the helical derivative corresponding to the spectral factorization of the nine-point Laplacian.
"""

# ╔═╡ 7f311104-b2f5-4082-865a-cb029fcc277a
md"""
## Spatial interpolation contest

We return to the spatial interpolation contest using rainfall data from Switzerland. We will perform iterative inverse interpolation using the formulation that involves the model covariance.

$\widehat{\mathbf{m}} = \mathbf{C}_{m}\,\mathbf{F}^T\,\left(\mathbf{F}\,\mathbf{C}_{m}\,\mathbf{F}^T + \mathbf{C}_n\right)^{-1}\,\mathbf{d}\;,$

where $\mathbf{d}$ represents rainfall measurements at the given weather stations, $\widehat{\mathbf{m}}$ is the estimated rainfall map, and $\mathbf{F}$ is the forward interpolation operator applied to the map.

We assume the noise is uniform and Gaussian: $\mathbf{C}_n = \epsilon^2\,\mathbf{I}$, and the model covariance can be decomposed as $\mathbf{C}_{m} = \mathbf{P}_m\,\mathbf{P}_m^T$, analogous to the previously used splitting of the inverse covariance $\mathbf{C}_m^{-1} = \mathbf{R}_m^T\,\mathbf{R}$.

If $\mathbf{R}_m$ is the gradient filter, then $\mathbf{C}_m^{-1}$ is the Laplacian, and $\mathbf{P}_m$ can be implemented as inverse (recursive) convolution using the helical derivative.
"""

# ╔═╡ 853f6363-7c11-4a5c-a67b-f42b76f83985
function conjgrad(forward::Function, adjoint::Function, precon::Function, 
                  d::Array, x0::Array, niter::Int)
    "Conjugate-gradients for minimizing |forward(precon(x))-d|^2"
 	x = deepcopy(x0)
    R = forward(precon(x, false)) - d  
    s, S = similar(x), similar(d)
    gnp = zero(eltype(x))
    for iter in 1:niter
        g = precon(adjoint(R), true)
        G = forward(precon(g, false))
        gn = g ⋅ g    
        if iter==1 # steepest descent
            s, S = g, G
        else            
            β = gn/gnp
            s = g + β*s
            S = G + β*S
        end
        gnp = gn      
        α = -gn/(S ⋅ S)
        x += α*s
        R += α*S
		@show iter, (R ⋅ R) 
    end
    return precon(x, false)
end

# ╔═╡ b47471c9-a7f1-4849-b853-06fcf838e8a8
begin
	# download data files
	download("https://ahay.org/data/rain/alldata.rsf@","alldata.bin")
	download("https://ahay.org/data/rain/obsdata.rsf@","obsdata.bin")
end

# ╔═╡ 01eb92e7-99fc-4570-8207-fbe8d48315a4
begin
	# read data
	alldata = Array{Float32}(undef, 3, 467); # single-precision array
	obsdata = Array{Float32}(undef, 3, 100); # single-precision array
	read!("alldata.bin", alldata)
	read!("obsdata.bin", obsdata)
end

# ╔═╡ 0a091dcb-7d40-43f8-828c-73afd494986e
function lint(regul::Array, coord; d=[1,1], o=[0,0])
    "bilinear interpolation"
    n = size(regul)
    nd = size(coord, 2)
    irreg = Array{eltype(regul)}(undef, nd)
    for id in 1:nd
		# find nearest neighbor
		x1 = 1 + (coord[1,id] - o[1])/d[1]
		x2 = 1 + (coord[2,id] - o[2])/d[2]
        i1, i2 = floor(Int, x1), floor(Int, x2)
		a1, a2 = x1 - i1, x2 - i2
		b1, b2 = 1 - a1, 1 - a2 
        if 0 < i1 && i1 < n[1] && 0 < i2 && i2 < n[2]
            irreg[id] = regul[i1,i2]*b1*b2 + regul[i1+1,i2]*a1*b2 +
			            regul[i1,i2+1]*b1*a2 + regul[i1+1,i2+1]*a1*a2
        end
    end
    return irreg
end

# ╔═╡ c60f5c19-a29e-4725-a8a0-fbe8ccb01d07
function lint_adjoint(irreg::Vector{T}, coord, n::Vector{Int}; 
                      d=[1,1], o=[0,0]) where T <: Real
    "adjoint of bilinear interpolation"
    nd = size(coord, 2)
    regul = zeros(T, n[1], n[2])
    for id in 1:nd
		# find nearest neighbor
		x1 = 1 + (coord[1,id] - o[1])/d[1]
		x2 = 1 + (coord[2,id] - o[2])/d[2]
        i1, i2 = floor(Int, x1), floor(Int, x2)
		a1, a2 = x1 - i1, x2 - i2
		b1, b2 = 1 - a1, 1 - a2 
        if 0 < i1 && i1 < n[1] && 0 < i2 && i2 < n[2]
			regul[i1,i2]     += irreg[id]*b1*b2
			regul[i1+1,i2]	 += irreg[id]*a1*b2
			regul[i1,i2+1]   += irreg[id]*b1*a2
			regul[i1+1,i2+1] += irreg[id]*a1*a2
        end
    end
    return regul
end

# ╔═╡ ef3dd2af-49c9-4c3b-a542-903c7bd5a047
begin
	lat = -185:185 # latitude
	lon = -127:127 # longitude
	nlat, nlon = length(lat), length(lon)
end

# ╔═╡ e4d776d8-d834-4aed-87e4-d1302428c167
begin
	forward(x) = lint(x, obsdata, o=[lat[1], lon[1]])
	adjoint(y) = lint_adjoint(y, obsdata, [nlat, nlon], o=[lat[1], lon[1]])
	# preconditioning
	precon(x, adj) = hrecursive(x, helder, adj)
end

# ╔═╡ 213a1957-a6da-42f5-8368-249c0fddc708
rain0 = zeros(Float32, 371, 255);

# ╔═╡ e5c266f8-c106-45ee-a980-199d96cbbed6
map10 = conjgrad(forward, adjoint, precon, obsdata[3,:], rain0, 10);

# ╔═╡ 0dd14e51-54c8-4755-908b-a052f10b6b4f
heatmap(lat, lon, map10', title="Helical Preconditioning (10 iterations)", cmap=:viridis)

# ╔═╡ a4286dc5-f957-48ec-9dc4-c268119a797d
predict = lint(map10, alldata, o=[lat[1], lon[1]]);

# ╔═╡ b6771e75-bc74-47d9-8465-3dae76d8be7e
function plot_corr(predict::Vector, title::String)
	lim = [-10,600]
	exact = alldata[3,:]
	# correlation coefficient
	cc = (exact ⋅ predict)/(sqrt(exact ⋅ exact) * sqrt(predict ⋅ predict))
	scatter(exact, predict, xlabel="True", ylabel="Predicted", 
		title="$title, cc=$(Float16(cc))",
	    aspect_ratio=:equal, xlim=lim, ylim=lim, label=:none)
	plot!(lim, lim, label=:none)
end

# ╔═╡ c1ba6693-877c-49f9-88b9-976cbfe506a7
plot_corr(predict, "Helical Preconditioning")

# ╔═╡ 0bc784ec-0b7e-4f7f-a498-6657c8cc67a6
md"""
!!! assignment

    ## Task 4

    Find experimentally the optimal number of iterations for this approach.
"""

# ╔═╡ 2562f16a-6a08-4efe-98d2-412d6bad0c84
md"""
!!! assignment

    ## Task 5

    Compare the result with the gradient regularization from the previous assignment in terms of accuracy and efficiency.
"""

# ╔═╡ 5a92aa15-c5e8-44f0-83c9-d3c50abc9147
md"""
!!! assignment

    ## Bonus Task

    Try improving the rainfall interpolation result by replacing the helical derivative with an alternative filter.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUIExtra = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
HTTP = "~1.11.0"
Plots = "~1.41.6"
PlutoUIExtra = "~0.1.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "23bda211212e72ba4264e8e1a0118ecfb24b58e4"

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

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataPipes]]
git-tree-sha1 = "3fb39158bc35c984cac5edb1ff55daa88a4b5074"
uuid = "02685ad9-2d12-40c3-9f73-c6aeda6a7ff5"
version = "0.3.19"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e86f4a2805f7f19bec5129bc9150c38208e5dc23"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.4"

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

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

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
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "f76f7560267b840e492180f9899b472f30b88450"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

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

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

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

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

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
# ╟─07e27cd5-4c74-4e73-b1a7-d5fe2651f02f
# ╟─0a8c1424-35f5-44f7-97bb-8ba02df829ec
# ╟─0b62a901-60f5-4eec-a5e3-e92f880ff6c6
# ╟─c8439cda-03b0-44a0-b5ad-14617f30ba32
# ╟─7beae512-736a-48ff-af57-41e7f805f599
# ╠═233d6871-5184-487a-b631-9b6de417eabd
# ╠═908b43bb-6980-42ef-8b58-d97e299ff66d
# ╠═ba18eae6-c56c-4838-85d5-2a37bdaba6c1
# ╠═4d409430-0f82-44df-ab59-1a9108f1e6d4
# ╠═868cfea3-4029-416a-95e3-81f4e05a7a81
# ╠═e0781ad3-5c46-4cb6-baac-5ab20a02dbfa
# ╠═c87909ed-82da-4c0b-9cd7-41b91166c693
# ╟─0a686f45-2554-4e2b-9b5f-27212ecc044c
# ╟─5327b862-bd75-4fad-93d0-c6898ea9c673
# ╟─00e9dff1-4d9c-4d66-ad0e-d33525646c9e
# ╟─a0d7a53c-a85c-4d94-bcef-925f1ada4138
# ╟─5ccde062-0cf9-4fcd-944f-850e80a81e46
# ╟─b308544d-342f-4aaa-9eb4-9e9d0e877645
# ╟─edc19bb9-8929-4117-b040-8cb9cee02a25
# ╟─157f5f20-58e3-41eb-97b9-f9ff590153b8
# ╠═0659951e-47c5-4115-b3c0-48ef4d5e1c15
# ╠═2fc1789d-bbf6-4689-8040-68277c5b5dc3
# ╠═2cd510b3-7d52-4fc2-b54e-8a584fb501f8
# ╠═a6277f35-419b-4c08-97bd-e68c668a9b5d
# ╟─f73c8d74-8c19-47b0-8548-aa5fc5fd8cdc
# ╠═105c82f6-ba75-4118-b4d2-6804796af16f
# ╠═9cad3b34-e5fb-4512-aa33-57341c14d06a
# ╠═96319f90-4af2-45d2-bc70-68e7b09da964
# ╠═0346e4ce-7150-4afa-b506-747ac6047d31
# ╠═cc9fd071-8c16-4e17-9614-a43045abb148
# ╠═f75dad75-f590-414b-8c9b-25050f225cd4
# ╠═cf0d841f-1dbb-4276-aad9-e8b3def57392
# ╠═c4d612dc-73fe-41ec-92e1-666fcc8f2662
# ╠═dc5e6482-1fb9-4830-8f30-9626118f02e6
# ╠═330404b7-6695-4585-bcf9-e9933900279f
# ╠═87bd88cd-d8a3-4d0f-927f-77f77c669fb1
# ╠═19b733ed-1d7f-4525-82de-7ac152e60190
# ╠═132b5edd-52e8-47a5-a68e-5f4448b9533d
# ╟─108e4423-f61f-4bbf-b053-fbea44eab5a7
# ╠═58b6375b-9067-4ee2-b260-4b0ae3b8e426
# ╠═13e36512-7ced-4a91-9e35-32b34ffd8cfa
# ╠═cc8a8ca1-2f66-4574-afbd-ab2649f3329f
# ╟─137d4887-fb9a-4c72-8318-b98c05143b41
# ╟─59f2dc5b-d9bb-41e6-b166-e33e976696d4
# ╠═801118cb-70da-4ffc-aca2-c41931337966
# ╠═278f7dea-9887-4dfc-af36-cea4ebc51039
# ╠═d2bef630-c655-4b96-8445-37e8e3c5e1db
# ╠═310173c7-f937-4411-b38d-578172009ec8
# ╠═eb6f694f-77d8-42bb-87c5-fa8c4eb0b865
# ╠═6a43dd30-7799-4509-a175-0459770f7488
# ╠═4ae0dd57-182c-47f0-8243-f0ed9f3c7ff9
# ╠═fb604622-7044-4564-8773-059d08e06715
# ╠═970126c6-8e4a-4b01-9f1a-80290e372f85
# ╠═fb5730c4-c250-4979-af8e-20c337a6393b
# ╟─a058667e-f3a8-47e5-83b1-73a8bf7680c4
# ╠═37282a3e-faf6-4dc1-be12-30dbb2e2189c
# ╠═538a1c40-047f-4624-9957-d7b3b2335005
# ╠═15f4a40c-366d-4e44-b490-33694f5749fa
# ╠═764abb6c-da30-4fe6-bba3-28e4c91a7b6f
# ╠═ca01a561-a004-4168-b4c0-414d74529b1f
# ╠═d52c06c2-2be5-489e-8582-3ef27d156c99
# ╠═16c3eca9-03f3-4d01-abdf-92c1b969da0d
# ╠═20a68a1e-02ce-4ba8-a227-eae93233fc74
# ╠═7367d669-656c-4333-bfc9-1222a89d93f1
# ╠═89eb1ba3-20c0-4c26-b8b6-47ab0c15c3d6
# ╠═ef0c5550-69e4-4848-806a-1394a096d57d
# ╠═fcc8b334-b706-4c52-b4bc-a01cbfd09ab1
# ╟─bc4cc41a-1208-4c7f-8cad-2131a0e192e5
# ╟─7f311104-b2f5-4082-865a-cb029fcc277a
# ╠═853f6363-7c11-4a5c-a67b-f42b76f83985
# ╠═b47471c9-a7f1-4849-b853-06fcf838e8a8
# ╠═01eb92e7-99fc-4570-8207-fbe8d48315a4
# ╠═0a091dcb-7d40-43f8-828c-73afd494986e
# ╠═c60f5c19-a29e-4725-a8a0-fbe8ccb01d07
# ╠═ef3dd2af-49c9-4c3b-a542-903c7bd5a047
# ╠═e4d776d8-d834-4aed-87e4-d1302428c167
# ╠═213a1957-a6da-42f5-8368-249c0fddc708
# ╠═e5c266f8-c106-45ee-a980-199d96cbbed6
# ╠═0dd14e51-54c8-4755-908b-a052f10b6b4f
# ╠═a4286dc5-f957-48ec-9dc4-c268119a797d
# ╠═b6771e75-bc74-47d9-8465-3dae76d8be7e
# ╠═c1ba6693-877c-49f9-88b9-976cbfe506a7
# ╟─0bc784ec-0b7e-4f7f-a498-6657c8cc67a6
# ╟─2562f16a-6a08-4efe-98d2-412d6bad0c84
# ╟─5a92aa15-c5e8-44f0-83c9-d3c50abc9147
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
