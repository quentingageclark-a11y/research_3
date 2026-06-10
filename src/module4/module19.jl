### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ 42e5f245-35fc-46db-b939-f695f51844f0
begin
	using PlutoUIExtra
	TableOfContents()
end

# ╔═╡ 8135e514-8c52-41f6-a025-28b772b6b979
using Plots

# ╔═╡ db4aca02-9ac2-438b-950f-6c1f517e3f28
using ZipFile

# ╔═╡ c963316b-9a9d-4d70-b2d8-4a04e7aae663
using Statistics

# ╔═╡ f42c3558-e304-449b-b732-064d4fc5cd0a
md"""
* Fomel, S. and Claerbout, J., 2024. Streaming prediction-error filters. Geophysics, 89(5), pp.F89-F95.
"""

# ╔═╡ f2ac8139-299b-4282-a484-645fe50fe87a
import HTTP

# ╔═╡ 6407e428-2790-4b14-9009-83e259c7eb8e
let
    notebook_dir = @__DIR__
    toc_path = joinpath(notebook_dir, "TOC_Notebook.jl")
    
    Markdown.parse("[⬅️ Back to Table of Contents](./open?path=$(HTTP.escapeuri(toc_path)))")
end

# ╔═╡ 2880810f-7d34-4961-a5d8-abeac8944b1c
md"""
# Streaming prediction-error filters

Prediction-error filters (PEFs) are essential in seismic deconvolution and other geophysical estimation problems. We show that non-stationary multidimensional PEFs can be computed in a "streaming" manner, where the filter gets updated incrementally by accepting one new data point at a time. The computational cost of estimating a streaming PEF reduces to the cost of a single convolution. In other words, the cost of PEF design while filtering equals the cost of applying the filter. Moreover, the non-linear operation of finding and applying a streaming PEF is invertible at a similar cost, which enables a fast approach to missing data interpolation.
"""

# ╔═╡ ee8414b6-2643-4f50-ab04-c775ecfbff53
md"""
## Introduction

Prediction-error filters (PEFs) play an essential role in geophysical data analysis. They can be used directly in different forms of seismic deconvolution (Webster, 1978; Robinson & Osman, 1996). More importantly, PEFs are a practical approximation for covariance operators used in geophysical estimation problems (Claerbout, 2014). As such, they provide a way to absorb and characterize statistical patterns in a given dataset (Claerbout & Brown, 1999).

Because most natural patterns are non-stationary, extending the classic stationary formulation to non-stationarity is vital. This extension can be done either by "patching" or breaking data into overlapping windows or by a smoothly non-stationary estimation with regularization (Crawley et al., 1999; Curry, 2003). Shaping regularization provides a particularly effective method for constraining the filter variability (Fomel, 2009; Liu & Fomel, 2011; Liu et al., 2012). Both approaches have an additional computational expense, particularly in storing variable filter coefficients (Ruan et al., 2015).

This paper presents an efficient approach to computing and applying non-stationary PEFs with no excessive storage requirements. Using an adaptive-filtering approach (Widrow \& Stearns, 1985; Haykin, 2002), We show that a non-stationary PEF can be updated on the fly by accepting one data point at a time. The cost of computing and applying such a PEF reduces to the cost of a single convolution. In other words, the cost of PEF design while filtering equals the cost of applying the filter. Moreover, the non-linear operation of finding and applying a PEF has an exact inverse, which finds an immediate application in missing data interpolation problems. We extend the
filter from 1-D to multiple dimensions using the helix transform (Claerbout, 1998) and show its application to simple benchmark problems.
"""

# ╔═╡ 9d04a877-45fe-4016-95a2-b41891dd2ddb
md"""
## Theory       

The basic equation defining a prediction-error filter is auto-regression. For a given data stream $\mathbf{d}$ and a one-dimensional PEF $\mathbf{a}$, the auto-regression equation takes the form

$$\left[\begin{array}{ccccc} d_{n+1} & d_n & d_{n-1} & \cdots &
                                                   d_{n+1-k} \end{array}\right]\,
\left[\begin{array}{l} 1 \\ a_1 \\ a_2 \\ \cdots \\ a_k \end{array}\right]
\approx 0\;,$$

where it is assumed that $n \ge k$. Equation 1 represents the convolution of the data with the prediction-error filter.

Suppose the filter gets updated with each new data point
$d_{n+1}$. The additional constraint we can impose to control
the variability of the filter is that the new filter $a$ stays close
to the previous filter $\bar{a}$.

The two conditions can be combined into one overdetermined linear system

$$\left[\begin{array}{cccc} d_n & d_{n-1} & \cdots & d_{n+1-k} \\
      \lambda & 0 & & 0 \\
      0 & \lambda & & 0 \\
    \cdots & \cdots & & \cdots \\
      0 & 0 & \cdots & \lambda
    \end{array}\right]\,
  \left[\begin{array}{l} a_1 \\ a_2 \\ \cdots \\ a_k \end{array}\right] \approx
  \left[\begin{array}{l} - d_{n+1} \\
      \lambda\,\bar{a}_1 \\
      \lambda\,\bar{a}_2 \\
     \cdots \\
      \lambda\,\bar{a}_k
    \end{array}\right]\;,$$

where $\lambda$ is the parameter that controls how much we allow 
$a$ to deviate from $\bar{a}$. In a shortened block-matrix notation, we can rewrite the linear system of equations as

$$\left[\begin{array}{c} \mathbf{d}^T \\ \lambda\,\mathbf{I} \end{array}\right]\,\mathbf{a} \approx
  \left[\begin{array}{l} - d_{n+1} \\
      \lambda\,\bar{\mathbf{a}} 
    \end{array}\right]\;,$$
    
where

$$\mathbf{d} = \left[\begin{array}{l} d_n \\ d_{n-1} \\ \cdots \\ d_{n+1-k} \end{array}\right]\;, \quad
\mathbf{a} = \left[\begin{array}{l} a_1 \\ a_2 \\ \cdots \\ a_k \end{array}\right]\;,$$

and $\mathbf{I}$ is the identity matrix. 

The least-squares solution of the overdetermined system is

$$\mathbf{a} = \left(\mathbf{d}\,\mathbf{d}^T + \lambda^2\,\mathbf{I}\right)^{-1}\,
  \left(-d_{n+1}\,\mathbf{d} + \lambda^2\,\bar{\mathbf{a}}\right)\;.$$

Next, we can use the Sherman-Morrison formula from linear algebra
(Hager, 1989) to transform the inverse matrix as follows:

$$\left(\mathbf{d}\,\mathbf{d}^T + \lambda^2\,\mathbf{I}\right)^{-1} =
  \frac{1}{\lambda^2}\,\left(\mathbf{I} - \frac{\mathbf{d}\,\mathbf{d}^T}{\lambda^2+\mathbf{d}^T\,\mathbf{d}}\right)\;.$$

Substituting the Sherman-Morrison equation and doing
algebraic simplifications lead to the final result:

$$\begin{array}{rcl}\mathbf{a} & = & \displaystyle \frac{1}{\lambda^2}\,\left(\mathbf{I} -
                   \frac{\mathbf{d}\,\mathbf{d}^T}{\lambda^2+\mathbf{d}^T\,\mathbf{d}}\right)\,\left(-d_{n+1}\,\mathbf{d}
                   + \lambda^2\,\bar{\mathbf{a}}\right) \\
& = &
\displaystyle \bar{\mathbf{a}} -
  \left(\frac{d_{n+1}+\mathbf{d}^T\,\bar{\mathbf{a}}}{\lambda^2+\mathbf{d}^T\,\mathbf{d}}\right)\,\mathbf{d}\;.\end{array}$$
  
The Sherman-Morrison technique is a well-known approach in adaptive filtering by recursive least-squares (Haykin, 2002).
  
This equation shows that the filter is updated by subtracting
a scaled version of the data. The scale is proportional to the
convolution residual computed using the previous version of the
filter. Updating the filter requires only elementary algebraic operations (vector dot products) and no iterations. Moreover, computing the dot product
$\mathbf{d}^T\,\mathbf{d}$ in a "streaming" fashion requires at most
two multiplications:

$$\mathbf{d}^T\,\mathbf{d} = \bar{\mathbf{d}}^T\,\bar{\mathbf{d}} +
d_n^2 - d_{n-k}^2\;,$$

where $k$ is number of points in $\mathbf{a}$ and $\mathbf{d}$.
"""

# ╔═╡ fa3791ea-ae2c-48ce-a2e9-c7589da2d7f2
md"""
### Inversion

The streaming residual $r_{n+1}$ from the left-hand side of
equation 1 can be expressed as

$$r_{n+1} = d_{n+1} + \mathbf{d}^T\,\mathbf{a}$$

or, substituting equation for $\mathbf{a}$,

$$\begin{array}{rcl}r_{n+1} & = & \displaystyle d_{n+1} + \mathbf{d}^T\,\left[\bar{\mathbf{a}} -
              \left(\frac{d_{n+1}+\mathbf{d}^T\,\bar{\mathbf{a}}}{\lambda^2+\mathbf{d}^T\,\mathbf{d}}\right)\,\mathbf{d}\right]
  \\
\nonumber
& = & \displaystyle \left(d_{n+1} +
      \mathbf{d}^T\,\bar{\mathbf{a}}\right)\,\left(1-\frac{\mathbf{d}^T\,\mathbf{d}}{\lambda^2+\mathbf{d}^T\,\mathbf{d}}\right)
  \\
& = & \displaystyle \frac{\lambda^2\,\left(d_{n+1} +
  \mathbf{d}^T\,\bar{\mathbf{a}}\right)}{\lambda^2+\mathbf{d}^T\,\mathbf{d}}\;.\end{array}$$


The equation for $r_{n+1}$ can be directly inverted to reconstruct~$d_{n+1}$ as follows:

$$d_{n+1} = r_{n+1}\,\left(1+\frac{\mathbf{d}^T\,\mathbf{d}}{\lambda^2}\right) - \mathbf{d}^T\,\bar{\mathbf{a}}\;.$$

We see that streaming time-variable deconvolution is invertible: both the input data and the time-varying filter can be reconstructed from the streaming residual. Note that because of the division by $\lambda^2$, the inverse operation may become numerically unstable for small $\lambda$.
"""

# ╔═╡ 11a5963a-f71e-46db-91c2-7809769cae00
begin
	n1=400

	# exponentially decaying sinusiods with different frequencies
	inp = zeros(Float32,n1)
	for j in n1÷15:n1÷5:n1
		wave = zeros(Float32,n1)
		for x in j:399
			y = (x-j)/400
			wave[x] = exp(-y*15)*sin(y*0.95*j)
		end
		inp += wave
	end
end

# ╔═╡ d7daeece-0c7a-429c-87c9-f898269b24f6
function stems(data, label, color) 
    "plot data using stems"
    plt=plot(zeros(Float32, n1), label=:none, color=:black)
    plot!(plt, data, line=:stem,
          label=label, color=color, legend=:outerleft, 
          xlim=[0.5, n1+0.5], border=:none)   
    return plt
end

# ╔═╡ 592d64b2-5e4a-436c-8f1a-896e364fa89d
function stream!(inv::Bool, d::Vector{T}, r::Vector{T}, na::Int, λ::Real) where T <: Real
    a = zeros(T, na) # streaming PEF
    dd = da = zero(T) # d (dot) d, d (dot) a
    for ia in 1:na
        if (inv) 
            d[ia] = r[ia]
        else 
            r[ia] = d[ia]
        end
        dd += d[ia]*d[ia]
    end
    for i1 in na+1:n1
        if (inv) # from r to d
            rn = r[i1] / λ 
            dn = rn * (λ + dd) - da
            d[i1] = dn
        else     # from d to r
            dn = d[i1]
            rn = (dn + da) / (λ + dd)
            r[i1] = λ * rn
        end
        # update PEF
        for ia in 1:na; a[ia] -= rn * d[i1-ia]; end
        # update dd and da
        dd += dn*dn - d[i1-na] * d[i1-na]   
        da = dn * a[1]
        for ia in 2:na; da += a[ia] * d[i1-ia+1]; end
    end
end

# ╔═╡ b754a2f3-9f21-4629-91a9-12d0ce91da88
begin
	res = similar(inp);
	bak = similar(inp);
	
	stream!(false, inp, res, 2, 0.1)
	stream!(true, bak, res, 2, 0.1)
	
	plot_input = stems(inp,"input",:blue);
	plot_decon = stems(res, "decon", :green);
	plot_inverse = stems(bak,"inverse",:purple);
	plot(plot_input, plot_decon, plot_inverse, layout=(3, 1))
end

# ╔═╡ 982f3cdb-580a-45de-97d0-c9453ec0921c
md"""
### Multiple dimensions

In order to extend the filter from 1-D to multiple dimensions, we follow the
helix transform of Claerbout (1998). The change is minimal:
the multidimensional data is arranged in a 1D stream on a helix, and
the definition of the data vector $\mathbf{d}$ changes to

$$\mathbf{d} = \left[\begin{array}{l} d_{n+1-l_1} \\ d_{n+1-l_2} \\ \cdots \\ d_{n+1-l_k} \end{array}\right]$$

where $l_1, l_2, \ldots, l_k$ represent lags of the helical filter. The computational cost and other benefits of streaming PEFs remain the same.

One caveat is that, in multiple dimensions, we may want the
nonstationary filter to change smoothly along the helix and other dimensions. As shown below, this goal can be accomplished with only a minor change to the algorithm. Suppose that $\bar{\mathbf{a}}_1$ is the previous filter in the first (helical) dimension, and $\bar{\mathbf{a}}_2$ is the previous filter in the second dimension. We can keep the newly estimated streaming PEF close to both
filters by changing the matrix equation to

$$\left[\begin{array}{c} \mathbf{d}^T \\ \lambda_1\,\mathbf{I} \\  \lambda_2\,\mathbf{I}  \end{array}\right]\,\mathbf{a} \approx
  \left[\begin{array}{l} - d_{n+1} \\
      \lambda_1\,\bar{\mathbf{a}}_1 \\ \lambda_2\,\bar{\mathbf{a}}_2
    \end{array}\right]\;,$$

where $\lambda_1$ and $\lambda_2$ control the filter variability in the two directions. The least-squares solution of this system is

$$\mathbf{a} = \left(\mathbf{d}\,\mathbf{d}^T + \lambda^2\,\mathbf{I}\right)^{-1}\,
  \left(-d_{n+1}\,\mathbf{d} + \lambda_1^2\,\bar{\mathbf{a}}_1 +
    \lambda_1^2\,\bar{\mathbf{a}}_2 \right)\;,$$

where $\lambda^2 = \lambda_1^2+\lambda_2^2$. The new equation
is equivalent to the previous equation with the simple substitution

$$\bar{\mathbf{a}} = \displaystyle \frac{\lambda_1^2\,\bar{\mathbf{a}}_1 +
    \lambda_2^2\,\bar{\mathbf{a}}_2}{\lambda_1^2+\lambda_2^2}\;.$$

Thus, the only change in the algorithm is an increase in the storage to keep track of both $\bar{\mathbf{a}}_1$ and $\bar{\mathbf{a}}_2$. The extension to more dimensions is analogous.
"""

# ╔═╡ aa78c06f-48c2-451a-9c17-4b54d4c1e862
md"""
### Cost comparison

Table 1 compares the computational cost of different approaches to computing prediction-error filters. The cost of streaming PEF is minimal and reduces to the cost of a single convolution. Streaming PEF can capture the input data's non-stationary character without storing multiple copies of the filter.

Moreover, all streaming computations are local, which makes them
particularly suitable for modern hardware accelerators such as
GPGPU (Sanders & Kandrot, 2010) or Intel Xeon Phi (Jeffers & Reinders, 2013).

| Method | Storage | Cost |
|:-------|:--------|:-----|
| Stationary PEF | $O(N_a)$ | $O(N_a^2\,N_d)$ |
| Non-stationary PEF | $O(N_a\,N_d)$ | $ O(N_a^2\,N_d)$ |
| Streaming PEF | $O(N_a)$ | $O(N_a\,N_d)$ |
"""

# ╔═╡ 0bbf3b89-2e29-48f3-8af6-15392bb1c57f
md"""
## Missing data interpolation

The existence of the exact inversion for the application of a streaming PEF allows for a straightforward approach to missing data interpolation. When reading data in a streaming fashion, every time we meet a missing data point $d_{n+1}$, we can replace its value by a
value computed from the residual. The residual value $r_{n+1}$ in this case can be set to zero or to a random number (white noise) with the expected variance of the residual. In the latter case, it is possible to generate multiple equiprobable distributions for the interpolation
result (Clapp, 2000).
"""

# ╔═╡ 4a391a03-3bb4-42ef-a49d-42205ae76073
begin
	inp2 = deepcopy(inp)
	known = ones(Bool,n1)
	
	# Cut holes in the data and create a mask
	for hole in (55, 153, 246, 301, 376)
	    inp2[hole:hole+20] .= 0
	    known[hole:hole+20] .= false
	end
end

# ╔═╡ f3c6343f-bdb2-4ffc-af84-31707f29180d
function stream_missing!(d::Vector{T}, k::Vector{Bool}, na::Int, λ::Real) where T <: Real
    a = zeros(T, na) # streaming PEF
    da = zero(T) # d (dot) a
    dd = zero(T) # d (dot) d
    for ia in 1:na
        dd += d[ia]*d[ia]
    end
    for i1 in na+1:n1
        if (k[i1]) # from d to r
            dn = d[i1]
            rn = (dn + da) / (λ + dd)
        else       # assume r=0
            dn = - da
            rn = zero(T)
            d[i1] = dn
        end
        # update PEF
        for ia in 1:na
            a[ia] -= rn * d[i1-ia]
        end
        # update dd and da
        dd += dn*dn - d[i1-na] * d[i1-na]   
        da = dn * a[1]
        for ia in 2:na
            da += a[ia] * d[i1-ia+1]
        end
    end
end

# ╔═╡ ee2e93bd-4177-4475-9fd0-97b9713b06d4
begin
	miss = deepcopy(inp2)
	stream_missing!(miss,known,2,0.05)
end

# ╔═╡ 032b94e1-baa6-4f40-be47-3bc1a032ca0e
begin
	plot_ideal = stems(inp,"ideal ",:blue);
	plot_hole = stems(inp2,"input ",:green);
	plot_interp = stems(miss,"filled",:purple);
	plot(plot_ideal, plot_hole, plot_interp, layout=(3, 1))
end

# ╔═╡ 255dee1f-1fdc-41e9-aeba-e90867c801da
md"""
A simple 1D interpolation example using data from Figure 1 is shown in Figure 2. As evident from the figure, such interpolation, while capable of picking the non-stationary data pattern, remains one-sided and may create discontinuities at the other side of the data gap. We can rerun the streaming PEF using the opposite directions and stack the results to help with this issue.
"""

# ╔═╡ 94984c4a-2516-4ccf-a4d5-c8bb54130731
# download data from a public server
download("https://zenodo.org/api/records/11099632/files-archive", "files.zip")

# ╔═╡ a798ff27-7af7-47d2-ab34-d645a0a3f5a9
begin
	# unzip the archive file
	r = ZipFile.Reader("files.zip")
	# make a dictionary of files for easy access
	patterns = Dict{String, IO}()
	for file in r.files
	    name = splitext(file.name)[1]
	    patterns[name] = file
	end
	# "wood" pattern
	wood = Array{Float32}(undef, 128, 128) # single-precision array
	read!(patterns["wood"], wood)
	# "herring" pattern
	herr = Array{Float32}(undef, 128, 128) # single-precision array
	read!(patterns["herr"], herr)
	# "seismic" pattern
	seis = Array{Float32}(undef, 250, 125) # single-precision array
	read!(patterns["seis"], seis)
	# normalize
	m = mean(seis)
	seis .-= m
	scale = std(wood)/std(seis)
	seis *= scale
end

# ╔═╡ 13a3cf0c-a57d-4ca1-9e19-57341c60d0d1
heatmap(wood,axis=nothing,legend=:none,color=:grays)

# ╔═╡ 2cfcae1c-1741-4979-af6c-9b698ec636a2
heatmap(herr,axis=nothing,legend=:none,color=:grays)

# ╔═╡ 2fc922aa-bca0-4f10-acd5-67ba54613f28
function punch_hole(data::Matrix{T}) where T <: Real
    # make an elliptical hole
    n1, n2 = size(data)
    hole = similar(data)
    mask = zeros(Bool, n1, n2)
    for i2 in 1:n2, i1 in 1:n1
        x = (i1-1)/n1 - 0.5
        y = (i2-1)/n2 - 0.3
        u =  x + y
        v = (x - y)/2
        if (u*u + v*v < 0.15)
            hole[i1,i2] = zero(T)
        else
            hole[i1,i2] = data[i1,i2]
            mask[i1,i2] = true
        end
    end
    return hole, mask
end

# ╔═╡ 59fb30b6-ef94-4311-97d5-b0da125aa0ce
whole, wmask = punch_hole(wood);

# ╔═╡ dfa007ca-22dd-4cc3-a27c-0394255e605c
function helix(lag::Vector{Tuple{T, T}}, ci::CartesianIndices) where T <: Integer
    "convert filter lags to helix lags for a given grid"
    # middle of the grid
    mid = CartesianIndex(Tuple(last(ci)) .÷ 2)
    # helix index of middle
    hmid = LinearIndices(ci)[mid]
    # from Cartesian shift to helix shift
    return LinearIndices(ci)[map(x -> CartesianIndex(x) + mid, lag)] .- hmid
end

# ╔═╡ a08819cc-19d5-4867-99f2-1cbde2096368
import Random

# ╔═╡ 7dd86911-7727-40dc-8efc-7d19e9e1d684
function stream_missing_helix!(d, k, 
                               lag::Vector{Tuple{I, I}},
                               λ::Real, std=0, seed=1) where I <: Integer
    "Fill missing data in multiple dimensions using streaming PEF on a helix"
    n1, na = length(d), length(lag)
    hlag = helix(lag, CartesianIndices(d))
    maxlag = maximum(hlag)
    T = eltype(d)
    a = zeros(T, na) # streaming PEF
    da = zero(T) # d (dot) a
    dd = zero(T) # d (dot) d
    for ia in 1:na
        dd += d[maxlag+1-hlag[ia]]^2
    end
    Random.seed!(seed)
    for i1 in maxlag+1:n1
        if (k[i1])
            dn = d[i1]
            rn = (dn + da) / (λ + dd)
        else # assume r_n is random
            rn = std * randn() / λ
            dn = rn * (λ + dd) - da
            d[i1] = dn
        end
        # update PEF
        for ia in 1:na
            a[ia] -= rn * d[i1-hlag[ia]]
        end
        # update dd and da
        dd += dn * dn - d[i1-maxlag] * d[i1-maxlag]   
        da = dn * a[1]
        for ia in 2:na
            da += a[ia] * d[i1+1-hlag[ia]]
        end
    end
end

# ╔═╡ bd7ba08c-656f-43e6-abfa-679d0125c925
begin
	# 11 x 11 PEF
	lag=[(x,0) for x in 1:5]
	for k in 1:10
	    lag = vcat(lag,[(x,k) for x in -5:5])
	end
end

# ╔═╡ 7a1c7936-6325-4ce5-bbb2-c365d096752b
function fill_hole(forward::Bool, hole, mask, pad::Integer, noise=0, seed=1)
    if forward
        holepad = hcat(zeros(Float32, size(hole, 1), pad), hole)
        maskpad = hcat(zeros(Bool, size(hole, 1), pad), mask)
        stream_missing_helix!(holepad, maskpad, lag, 1e6, noise, seed)
        return holepad[:,pad+1:end]
    else
        rhole = reverse(hole)
        rmask = reverse(mask)
        holepad = hcat(zeros(Float32, size(rhole, 1), pad), rhole)
        maskpad = hcat(zeros(Bool, size(rhole, 1), pad), rmask)
        stream_missing_helix!(holepad, maskpad, lag, 1e6, noise, seed+1)
        return reverse(holepad[:,pad+1:end])
    end
end

# ╔═╡ b65f5893-4e12-40fd-8bdc-e1322bd97021
begin
	filled1 = fill_hole(true, whole, wmask, 20);
	filled2 = fill_hole(false, whole, wmask, 20);
end

# ╔═╡ 998901b2-a9a1-4e1d-94c4-7151daf08907
plot2(data, title) = heatmap(data, axis=nothing, yflip=:true, clim=(-137, 137),
                             legend=:none, color=:grays, title=title)

# ╔═╡ 8bf1145e-f2e9-43a6-bab0-36884d3b19b7
begin
	p1 = plot2(filled1, "(a) Filled 1")
	p2 = plot2(filled2, "(a) Filled 2")
	p3 = plot2(filled1 + filled2 - whole, "(c) Filled 1+2")
	plot(p1, p2, p3, layout=(1, 3))
end

# ╔═╡ 82217839-e72e-47b3-8647-d5bba16828ec
filled = fill_hole(true,  whole, wmask, 20, 2) + 
         fill_hole(false, whole, wmask, 20, 2) - whole

# ╔═╡ 60447c4a-046d-40f0-9361-a845d08b6602
begin
	pw1 = plot2(wood, "(a) Ideal") 
	pw2 = plot2(whole, "(b) Gapped")
	pw3 = plot2(filled, "(c) Filled")
	plot(pw1, pw2, pw3, layout=(1, 3))
end

# ╔═╡ da6ce3b8-d493-4c9f-ad2a-b29dc9775c35
md"""
Figure 3 shows a 2D missing data interpolation test using a synthetic pattern from  Claerbout & Brown (1999). The input data for this test is shown in Figure 4b. Interpolation proceeds by running the streaming PEF twice in the forward and backward directions (Figures 3a and 3b) and averaging the interpolation results (Figure 3c.) The cost of such procedure is the cost of two convolutions, as opposed to multiple iterations required in the conventional approach to missing data interpolation with PEFs  (Naghizadeh & Sacchi, 2010; Liu & Fomel, 2011; Claerbout, 2014). A better interpolation result in Figure 4c is achieved by filling the residual inside the gap with small-variance white noise instead of zeros while reconstructing
the data from the residual. 
"""

# ╔═╡ e0c8943e-43b7-46c8-b6e3-5a76d266bb21
begin
	p = Array{Plots.Plot}(undef,3)
	for k in 1:3
	    filled = fill_hole(true,  whole, wmask, 20, 2, k) + 
	             fill_hole(false, whole, wmask, 20, 2, k+3) - whole
	    p[k] = plot2(filled, "Realization $k") 
	end
	plot(p[1], p[2], p[3], layout=(1, 3))
end

# ╔═╡ 18f29fca-a750-41fa-9f7d-3cac74b6a4ef
md"""
The random residual approach can generate multiple equiprobable realizations for the missing data interpolation problem in the spirit of geostatistical stochastic simulations (Clapp, 2000). Figure 5 shows three realizations created using different seeds for the pseudorandom number generator. 
"""

# ╔═╡ b9c249c3-1205-4c5e-ac67-c1d19399218b
hhole, hmask = punch_hole(herr);

# ╔═╡ e174f1cb-ae7f-4423-a73b-8c7d426dc74e
hfilled = fill_hole(true,  hhole, hmask, 20, 6) + 
         fill_hole(false, hhole, hmask, 20, 6) - hhole;

# ╔═╡ 5e9f279b-c498-48c0-a291-25f665640497
begin
	ph1 = plot2(herr, "(a) Ideal") 
	ph2 = plot2(hhole, "(b) Gapped")
	ph3 = plot2(hfilled, "(c) Filled")
	plot(ph1, ph2, ph3, layout=(1, 3))
end

# ╔═╡ 2fc3dba3-26ce-4453-9977-2b7b5b74612a
md"""
Figure 6 shows a similar interpolation test applied to 2D data with a non-stationary pattern. Although streaming PEF fails to achieve a perfect reconstruction in this case, it manages to capture most of the pattern and hide the location of the gap.
"""

# ╔═╡ 2b6744c6-b63a-4a0e-a0df-0c2a088f20d9
# ╠═╡ disabled = true
#=╠═╡
using Statistics
  ╠═╡ =#

# ╔═╡ 72643e87-6194-49b7-b97d-84ba94f47491
shole, smask = punch_hole(seis);

# ╔═╡ 12f83bd0-2043-4ab3-b1a3-8f23871ea8c1
sfilled = fill_hole(true,  shole, smask, 20, 0.7) + 
         fill_hole(false, shole, smask, 20, 0.7) - shole;

# ╔═╡ 61457211-736c-45f2-8b91-0611ef051e4e
begin
	ps1 = plot2(seis, "(a) Ideal") 
	ps2 = plot2(shole, "(b) Gapped")
	ps3 = plot2(sfilled, "(c) Filled")
	plot(ps1, ps2, ps3, layout=(1, 3))
end

# ╔═╡ 93371886-bd2d-4b75-b594-f7b3342f2e73
md"""
Figure 7 shows a similar test applied to 2D seismic data. The pattern made by reflection events with variable slopes is well captured.
"""

# ╔═╡ ffcd9369-7080-4213-b4a1-6017885ca439
function stream_helix!(inv::Bool, d, r, lag::Vector{Tuple{I, I}}, λ::Real) where I <: Integer
    n1, na = length(d), length(lag)
    hlag = helix(lag, CartesianIndices(d))
    maxlag = maximum(hlag)
    T = eltype(d)
    a = zeros(T, na) # streaming PEF
    for i1 in 1:maxlag
        if (inv) 
            d[i1] = r[i1]
        else 
            r[i1] = d[i1]
        end
    end
    da = zero(T) # d (dot) a
    dd = zero(T) # d (dot) d
    for ia in 1:na      
        dd += d[maxlag+1-hlag[ia]]^2
    end
    for i1 in maxlag+1:n1
        if (inv) 
            rn = r[i1] / λ 
            dn = rn * (λ + dd) - da
            d[i1] = dn
        else 
            dn = d[i1]
            rn = (dn + da) / (λ + dd)
            r[i1] = λ * rn
        end
        # update PEF
        for ia in 1:na
            a[ia] -= rn * d[i1-hlag[ia]]
        end
        # update dd and da
        dd += dn * dn - d[i1-maxlag] * d[i1-maxlag]    
        da = dn * a[1]
        for ia in 2:na
            da += a[ia] * d[i1+1-hlag[ia]]
        end
    end
end

# ╔═╡ 913be130-0be1-4ebb-a3da-c2774f0789fa
begin
	# apply helix filter
	spad = hcat(zeros(Float32, size(seis, 1), 20), seis)
	sres= similar(spad)
	stream_helix!(false, spad, sres, lag, 1e6) # pad -> res
	stream_helix!(true,  spad, sres, lag, 1e6) # pad <- res
end

# ╔═╡ 54ed56f2-2ee0-42b0-8e30-9c0c7288a8c8
begin
	pss1 = plot2(seis, "(a) Input") 
	pss2 = plot2(20*sres[:,21:end], "(b) Residual (x 20)")
	pss3 = plot2(spad[:,21:end], "(c) Inverse")
	plot(pss1, pss2, pss3, layout=(1, 3))
end

# ╔═╡ 80efb290-d49d-421f-99dc-d98899d08ae3
md"""
In this case, removing dominant reflection events by PEF filtering leaves behind weaker hyperbolic diffraction events (Figure 8). 
"""

# ╔═╡ 5b85c76a-9cc0-4a96-8f56-9a3aee8570b6
md"""
## Discussion

A streaming application, where the adaptive filter is updated one data point at a time, is appropriate in a continuous stream of large amounts of data, such as in passive monitoring of carbon storage (Alumbaugh et al., 2024). In more typical scenarios, where the data are immediately accessible, more accurate results can be achieved with other forms of regularization, such as regularizing adaptive filter coefficients with smoothing shaping operators (Fomel, 2009; Liu \& Fomel, 2011; Liu et al., 2012). For greater efficiency, a hybrid regularization strategy can be developed. Such strategies are discussed by Geng et al. (2024), who also extend the streaming approach to other applications of seismic attributes.

In applications such as missing data reconstruction, the streaming approach represents an extreme point of the accuracy-efficiency trade-off. To achieve better accuracy, some of the efficiency can be sacrificed at the expense of performing extra iterations. The presented approach is also limited to situations of missing values in regularly sampled data and will need to be modified for situations of irregular data.

In multidimensional applications, the helical boundary conditions allow for easy invertibility (Claerbout, 1998) but are not always appropriate. In this case, extra accuracy can also be bought at the cost of sacrificing some of the efficiency.

We hope that providing reproducible benchmarks with this paper will invite other researchers to make direct comparisons with more advanced methods. Such comparisons are beyond the scope of this paper because they would require a different codebase.
"""

# ╔═╡ bd9089fc-b1ba-46b5-bfbb-16834caf0dd6
md"""
## Conclusions

We have presented an efficient approach to computing and applying non-stationary prediction-error filters (PEFs). Instead of storing multiple copies of varying filters, the streaming approach stores only one copy and updates the filter on the fly with every new data point. The cost of this procedure is equivalent to the cost of a single convolution and does not require multiple iterations. Moreover, the non-linear operation of estimating and applying a streaming PEF has an exact inverse, which becomes helpful in missing data interpolation
problems. A streaming approach to missing data interpolation does not require iterations and can be accomplished effectively at the cost of two convolutions. We anticipate many possible applications of the proposed technique in geophysical estimation problems.
"""

# ╔═╡ 32d15680-ba4b-4362-9780-84e048d36778
md"""
## References

Alumbaugh, D.L., Correa, J., Jordan, P., Petras, B., Chundur, S. and Abriel, W., 2024. An assessment of the role of geophysics in future US geologic carbon storage projects. The Leading Edge, 43, 72-83.

Bezanson, J., A. Edelman, S. Karpinski, and V. B. Shah, 2017, Julia: A fresh approach to numerical computing: SIAM Review, 59, 65-98.

Claerbout, J., 1998, Multidimensional recursive filters via a helix: Geophysics, 63, 1532–1541.

Claerbout, J., and M. Brown, 1999, Two-dimensional textures and prediction-error filters: 61st Mtg., Eur. Assn. Geosci. Eng., Session:1009.

Claerbout, J. F., 2014, Geophysical image estimation by example: Environmental soundings image enhancement: Lulu. (http://sep.stanford.edu/sep/prof/).

Clapp, R., 2000, Multiple realizations using standard inversion techniques, in SEP-105: Stanford Exploration Project, 67–78.

Crawley, S., J. Claerbout, and R. Clapp, 1999, Interpolation with smoothly non-stationary prediction-error filters: 69th Annual International Meeting, SEG, Expanded Abstracts, 1154–1157.

Curry, W., 2003, Interpolation of irregularly sampled data with nonstationary, multi- scale prediction-error filters: 73th Annual International Meeting, SEG, Expanded Abstracts, 1913–1916.

Fomel, S., 2009, Adaptive multiple subtraction using regularized nonstationary regression: Geophysics, 74, V25–V33.

Geng, Z., Fomel, S., Liu, Y., Wang, Q., Zheng, Z. and Chen, Y., 2024. Streaming seismic attributes. Geophysics, 89, A7-A10.

Granger, B.E. and F. P\'{e}rez, F., 2021, Jupyter: Thinking and storytelling with code and data: Computing in Science \& Engineering, 23, 7-14.

Hager, W. W., 1989, Updating the inverse of a matrix: SIAM Review, 31, 221–239. Jeffers, J., and J. Reinders, 2013, Intel Xeon Phi coprocessor high-performance programming: Morgan Kaufmann.

Haykin, S., 2002. Adaptive Filter Theory. Prentice-Hall.

Liu, G., X. C. J. Du, and K. Wu, 2012, Random noise attenuation using f-x regularized nonstationary autoregression: Geophysics, 77, V61–V69.

Liu, Y., and S. Fomel, 2011, Seismic data interpolation beyond aliasing using regularized nonstationary autoregression: Geophysics, 76, V69–V77.

Naghizadeh, M., and M. D. Sacchi, 2010, Robust reconstruction of aliased data using autoregressive spectral estimates: Geophysical Prospecting, 58, 1049–1062.

Robinson, E. A., and O. M. Osman, eds., 1996, Deconvolution 2: Soc. of Expl. Geophys.

Ruan, K., J. Jennings, E. Biondi, R. G. Clapp, S. A. Levin, and J. Claerbout, 2015, Industrial scale high-performance adaptive filtering with PEF applications, in SEP-160: Stanford Exploration Project, 177–188.

Sanders, J., and E. Kandrot, 2010, CUDA by example: An introduction to General- Purpose GPU programming: Addison-Wesley Professional.

Webster, G. M., ed., 1978, Deconvolution: Soc. of Expl. Geophys.

Widrow, B., and S.D. Stearns, Adaptive Signal Processing: Prentice Hall, 1985.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUIExtra = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
ZipFile = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"

[compat]
HTTP = "~1.10.15"
Plots = "~1.40.9"
PlutoUIExtra = "~0.1.8"
ZipFile = "~0.10.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "6ae75e785a76a3f48e8cb55942ac07fdb0f85b96"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "856ecd7cebb68e5fc87abecd2326ad59f0f911f3"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.43"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+4"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "c7acce7a7e1078a20a285211dd73cd3941a871d6"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.0"
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
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
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
git-tree-sha1 = "f36e5e8fdffcb5646ea5da81495a5a7566005127"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.3"

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
git-tree-sha1 = "29077a8d5c093f4e0988e92c0d76f56c4c581900"
uuid = "02685ad9-2d12-40c3-9f73-c6aeda6a7ff5"
version = "0.3.18"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

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
git-tree-sha1 = "e51db81749b0777b2147fbe7b783ee79045b8e99"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+3"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

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
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "424c8f76017e39fdfcdbb5935a8e6742244959e8"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "b90934c8cb33920a8dc66736471dc3961b42ec9f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "c67b33b085f6e2faf8bf79a61962e7339a81129c"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.15"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

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
git-tree-sha1 = "d966f85b3b7a8e49d034d27a189e9a4874b4391a"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.13"
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
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "71b48d857e86bf7a1838c4736545699974ce79a2"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.9"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "df37206100d39f79b3376afb6b9cee4970041c61"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.1+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "61dfdba58e585066d8bce214c5a51eaa0539f269"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89211ea35d9df5831fca5d33552c02bd33878419"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e888ad02ce716b319e6bdb985d2ef300e7089889"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.3+0"

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
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "926c6af3a037c68d02596a44c22ec3595f5f760b"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

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
git-tree-sha1 = "030ea22804ef91648f29b7ad3fc15fa49d0e6e71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ed6834e95bd326c52d5675b4181386dfbe885afb"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.55.5+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

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
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "dae01f8c2e069a683d3a6e17bbae5070ab94786f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.9"

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
git-tree-sha1 = "3ac7038a98ef6977d44adeadc73cc6f596c08109"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.79"

[[deps.PlutoUIExtra]]
deps = ["AbstractPlutoDingetjes", "ConstructionBase", "FlexiMaps", "HypertextLiteral", "InteractiveUtils", "IntervalSets", "Markdown", "PlutoUI", "Random", "Reexport"]
git-tree-sha1 = "b4ff5d24e2dc8fbf319cd44f9f81b5356e27bafb"
uuid = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"
version = "0.1.8"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

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
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

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
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

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
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

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
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

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

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

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
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "beef98d5aad604d9e7d60b2ece5181f7888e2fd6"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+3"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e9216fdcd8514b7072b43653874fd688e4c6c003"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.12+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89799ae67c17caa5b3b5a19b8469eeee474377db"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.5+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c57201109a9e4c0585b208bb408bc41d205ac4e9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.2+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6dba04dbfb72ae3ebe5418ba33d087ba8aa8cb00"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.1+0"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "622cf78670d067c738667aaa96c553430b65e269"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0ba42241cb6809f1a278d0bcb976e0483c3f1f2d"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+1"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7b5bbf1efbafb5eca466700949625e07533aff2"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.45+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╟─f42c3558-e304-449b-b732-064d4fc5cd0a
# ╟─42e5f245-35fc-46db-b939-f695f51844f0
# ╟─f2ac8139-299b-4282-a484-645fe50fe87a
# ╟─6407e428-2790-4b14-9009-83e259c7eb8e
# ╟─2880810f-7d34-4961-a5d8-abeac8944b1c
# ╟─ee8414b6-2643-4f50-ab04-c775ecfbff53
# ╟─9d04a877-45fe-4016-95a2-b41891dd2ddb
# ╟─fa3791ea-ae2c-48ce-a2e9-c7589da2d7f2
# ╠═11a5963a-f71e-46db-91c2-7809769cae00
# ╠═8135e514-8c52-41f6-a025-28b772b6b979
# ╠═d7daeece-0c7a-429c-87c9-f898269b24f6
# ╠═592d64b2-5e4a-436c-8f1a-896e364fa89d
# ╠═b754a2f3-9f21-4629-91a9-12d0ce91da88
# ╟─982f3cdb-580a-45de-97d0-c9453ec0921c
# ╟─aa78c06f-48c2-451a-9c17-4b54d4c1e862
# ╟─0bbf3b89-2e29-48f3-8af6-15392bb1c57f
# ╠═4a391a03-3bb4-42ef-a49d-42205ae76073
# ╠═f3c6343f-bdb2-4ffc-af84-31707f29180d
# ╠═ee2e93bd-4177-4475-9fd0-97b9713b06d4
# ╠═032b94e1-baa6-4f40-be47-3bc1a032ca0e
# ╟─255dee1f-1fdc-41e9-aeba-e90867c801da
# ╠═db4aca02-9ac2-438b-950f-6c1f517e3f28
# ╠═94984c4a-2516-4ccf-a4d5-c8bb54130731
# ╠═c963316b-9a9d-4d70-b2d8-4a04e7aae663
# ╠═a798ff27-7af7-47d2-ab34-d645a0a3f5a9
# ╠═13a3cf0c-a57d-4ca1-9e19-57341c60d0d1
# ╠═2cfcae1c-1741-4979-af6c-9b698ec636a2
# ╠═2fc922aa-bca0-4f10-acd5-67ba54613f28
# ╠═59fb30b6-ef94-4311-97d5-b0da125aa0ce
# ╠═dfa007ca-22dd-4cc3-a27c-0394255e605c
# ╠═a08819cc-19d5-4867-99f2-1cbde2096368
# ╠═7dd86911-7727-40dc-8efc-7d19e9e1d684
# ╠═bd7ba08c-656f-43e6-abfa-679d0125c925
# ╠═7a1c7936-6325-4ce5-bbb2-c365d096752b
# ╠═b65f5893-4e12-40fd-8bdc-e1322bd97021
# ╠═998901b2-a9a1-4e1d-94c4-7151daf08907
# ╠═8bf1145e-f2e9-43a6-bab0-36884d3b19b7
# ╠═82217839-e72e-47b3-8647-d5bba16828ec
# ╠═60447c4a-046d-40f0-9361-a845d08b6602
# ╟─da6ce3b8-d493-4c9f-ad2a-b29dc9775c35
# ╠═e0c8943e-43b7-46c8-b6e3-5a76d266bb21
# ╟─18f29fca-a750-41fa-9f7d-3cac74b6a4ef
# ╠═b9c249c3-1205-4c5e-ac67-c1d19399218b
# ╠═e174f1cb-ae7f-4423-a73b-8c7d426dc74e
# ╠═5e9f279b-c498-48c0-a291-25f665640497
# ╟─2fc3dba3-26ce-4453-9977-2b7b5b74612a
# ╠═2b6744c6-b63a-4a0e-a0df-0c2a088f20d9
# ╠═72643e87-6194-49b7-b97d-84ba94f47491
# ╠═12f83bd0-2043-4ab3-b1a3-8f23871ea8c1
# ╠═61457211-736c-45f2-8b91-0611ef051e4e
# ╟─93371886-bd2d-4b75-b594-f7b3342f2e73
# ╠═ffcd9369-7080-4213-b4a1-6017885ca439
# ╠═913be130-0be1-4ebb-a3da-c2774f0789fa
# ╠═54ed56f2-2ee0-42b0-8e30-9c0c7288a8c8
# ╟─80efb290-d49d-421f-99dc-d98899d08ae3
# ╟─5b85c76a-9cc0-4a96-8f56-9a3aee8570b6
# ╟─bd9089fc-b1ba-46b5-bfbb-16834caf0dd6
# ╟─32d15680-ba4b-4362-9780-84e048d36778
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
