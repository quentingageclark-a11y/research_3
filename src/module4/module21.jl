### A Pluto.jl notebook ###
# v0.20.27

#> [frontmatter]
#> chapter = 4
#> section = 22
#> order = 22
#> image = "https://raw.githubusercontent.com/fonsp/Pluto.jl/580ab811f13d565cc81ebfa70ed36c84b125f55d/demo/plutodemo.gif"
#> title = "Local Attributes"
#> tags = ["module4", "track_julia", "track_material", "Pluto", "PlutoUI"]
#> layout = "layout.jlhtml"

using Markdown
using InteractiveUtils

# ╔═╡ 1b9d9a5e-fef4-41d3-aede-320bbbd2fd4b
begin
	using PlutoUIExtra
	TableOfContents()
end

# ╔═╡ 2f845a97-486e-4daa-a88b-a784ee4e52eb
using Plots, Statistics

# ╔═╡ 3a80d65d-b1ff-4366-b058-cb1096d5574d
using LinearAlgebra # defines dot-product as "⋅"

# ╔═╡ c24ed93d-fd1c-41dd-8fdf-e542bd8b0ea2
using LocalSignalAttributes

# ╔═╡ 77fb049f-f800-4ccc-a662-b0ab59cc3a2b
using SegyIO # for reading data in sgy format

# ╔═╡ ad03ff21-1a42-45af-8036-54a22efe9bcb
using FFTW

# ╔═╡ 7edeb330-aa60-4757-9fcd-57ce154281e1
import HTTP

# ╔═╡ 8d4c7897-54ce-4d98-b3d9-eeff291ebdcc
let
    notebook_dir = @__DIR__
    toc_path = joinpath(notebook_dir, "TOC_Notebook.jl")
    
    Markdown.parse("[⬅️ Back to Table of Contents](./open?path=$(HTTP.escapeuri(toc_path)))")
end

# ╔═╡ 48e11ea3-d612-4822-b849-cba6b86f6fc5
md"""
# Local attributes

We revisit the definition of simple data attributes to understand how to transform them from global to localized measures using linear estimation tools and shaping regularization.
"""

# ╔═╡ 2aecc41f-6276-436b-91fb-65137ddcef54
md"""
## From global to local

Suppose we have two data vectors, $\mathbf{a}$ and $\mathbf{b}$, and want to find a scale factor $\gamma$ that would make $\mathbf{b}$ closer to $\mathbf{a}$ after scaling. The least-squares minimization

$$\displaystyle \min_{\gamma} \left|\mathbf{a} -
     \gamma\,\mathbf{b}\right|^2 = \displaystyle \min_{\gamma}
   \sum_n (a_n - \gamma\,b_n)^2$$

has the solution
 
$$\widehat{\gamma} =
   \displaystyle
   \frac{\mathbf{b}^T\,\mathbf{a}}{\mathbf{b}^T\,\mathbf{b}} =
   \displaystyle \frac{\sum_n a_n\,b_n}{\sum_n b_n^2}\;.$$

What if we want the scaling to be not a constant but a variable vector $\mathbf{c}$ that would scale different parts of the vector $\mathbf{b}$ differently for a closer match to the target? One approach would be to split the data into overlapping windows or patches, apply different scaling in each patch, and then stitch the patches back together. While this approach has some advantages, such as the ability to process different patches in parallel, it provides only crude control over localization.

To make the problem genuinely localized, we can consider the least-squares problem
 
$$\displaystyle \min_{\mathbf{c}} \left|\mathbf{a} -
     \mathbf{C}\,\mathbf{b}\right|^2 = \min_{\mathbf{c}} \left|\mathbf{a} -
     \mathbf{B}\,\mathbf{c}\right|^2 \displaystyle = \min_{\mathbf{c}}
   \sum_n (a_n - c_n\,b_n)^2\;,$$

where $\mathbf{C}=\mbox{diag}(\mathbf{c})$ and $\mathbf{B}=\mbox{diag}(\mathbf{b})$ represent diagonal matrices.

The problem is poorly defined because it may lead to unstable division when some elements of $b_n$ are small or zero.
"""

# ╔═╡ 48009579-80cd-44a8-ba01-62d847f91a82
md"""
We can constrain the unstable division by using regularization - for example, by enforcing the estimated vector $\mathbf{c}$ to be smooth. The solution with enforced smoothness, achieved through shaping regularization, is
 
$$\widehat{\mathbf{c}} = \left[\lambda^2\,\mathbf{I} + 
      \mathbf{S}\,\left(\mathbf{B}^2 - \lambda^2\,\mathbf{I}\right)\right]^{-1}\,
    \mathbf{S}\,\mathbf{B}\,\mathbf{a}\;,$$

where $\lambda$ is a scaling factor, which we can set to the root-mean-square value of  $b_n$:

$$\lambda = \displaystyle \sqrt{\frac{1}{N} \sum_{n=1}^N b_n^2}\;,$$

and $\mathbf{S}$ represents *model shaping*: a smoothing filter with the user-specified degree of smoothness. 

Efficient smoothing by triangle filtering is appropriate for this task. An iterative method, like conjugate gradients, can perform the inversion. The greater the smoothness, the closer the inverted matrix will be to the identity, requiring fewer iterations to reach convergence. With a very large smoothing radius, the solution $\widehat{\mathbf{c}}$ will start to approach the constant $\widehat{\gamma}$ case. Less smoothness enables the desired variability in localized scaling while maintaining stability in the inversion.
"""

# ╔═╡ 4f346e49-8555-4629-a4e9-a1c10ec91636
md"""
* Fomel, S., 2007, Local seismic attributes: Geophysics, 72, A29–A33.
"""

# ╔═╡ 25c51cfe-fe0f-468a-9f9d-e99a39aca7c6
md"""
### Automatic gain control

By setting the vector $\mathbf{a}$ to a constant, we can determine a weighting factor needed to equalize the values in the input data $\mathbf{b}$. This method is commonly known as AGC (automatic gain control).

The figure above provides an example. A raw input shot gather has amplitudes that decrease over time due to the combined effects of geometrical spreading and attenuation. By setting $\mathbf{a}$ to one and $\mathbf{b}$ to the absolute value of the data, we can determine the gain function $\mathbf{c}$ using the shaping-regularization method from the previous section. In this case, the smoothing radius is wide in the horizontal direction, so the necessary variable scaling applies only vertically. The result shows the expected increase over time. Applying the gain to the data results in more balanced amplitudes.
"""

# ╔═╡ d86563fd-98e3-44d0-be77-071a94241f5e
download("https://ahay.org/data/wz/data.rsf@","data.bin")

# ╔═╡ ff4dfd2f-fc26-411d-9968-ce8541d9f082
begin
	shot = Array{Float32}(undef, 2000, 81); # single-precision array
	read!("data.bin", shot)
	shot *= 1.0e-8 # normalize
end

# ╔═╡ 43495c00-e48c-43e7-9281-743342b2597e
begin
	t = range(0, 2, 1001) # time axis
	x = range(-2, 2, 81)  # offset axis
end

# ╔═╡ 6da8c15c-cb31-4a7e-849d-40e48cd1630e
function plot_seismic(data, title)
    clip = quantile(abs.(data[:]), 0.99)
    return heatmap(x, t, data[1:1001,:], clim=(-clip, clip), 
                   yflip=true, c=:grays, legend=:none,
                   xlabel="Offset (km)", ylabel="Time (s)", 
                   title=title)
end

# ╔═╡ 58c1b007-7fa2-470a-a82f-f968ec48ff3f
size(shot), size(t), size(x)

# ╔═╡ fc5b3ccd-8aed-4930-ab9d-5d8b057d72ba
pshot = plot_seismic(shot, "(a) Original data")

# ╔═╡ 6486abf2-97ea-4a8d-b98a-1ea4ed9168db
function conjgrad(forward::Function, adjoint::Function, shaping::Function, ϵ::Float64, d::Array, p0::Array, niter::Int; tol=1.0e-6)
    "Conjugate-gradient algorithm for shaping regularization"
    p = deepcopy(p0)
    x = shaping(p)
    r = forward(x) .- d      
    sp, sx, sr = similar(p), similar(x), similar(r)
    g0 = gnp = zero(eltype(x))
    for iter in 1:niter
        gx = adjoint(r) - ϵ*x
        gp = shaping(gx) + ϵ*p
        gx = shaping(gp)
        gr = forward(gx)
        
        gn = gp ⋅ gp
        @show iter, gn 
        
        if iter==1
            sp, sx, sr = gp, gx, gr
			g0 = gn
        else
            β = gn/gnp
			γ = gn/g0
			if (β < tol || gn/g0 < tol)
				println("converged: β=$(β) γ=$(γ)")
				break
			end
            sp = gp + β*sp
            sx = gx + β*sx
            sr = gr + β*sr
        end
        gnp = gn
        
        α = sr ⋅ sr + ϵ*(sp ⋅ sp -  sx ⋅ sx)
        α = - gn/α
        
        p = p + α*sp
        x = x + α*sx
        r = r + α*sr
    end
    return x
end

# ╔═╡ 2dfa5cf8-198a-497c-8542-fe5e432df176
md"""
!!! assignment
    ## Task 1

    Replace the absolute value in the AGC implementation with the envelope and compare the results.
"""

# ╔═╡ 53401cc8-9710-4d00-bdff-ec0e3655165e
# The envelope is computed with the help of the Hilbert transform
hilb = mapslices(hilbert, shot, dims=1);

# ╔═╡ 2117d334-20e7-45ca-b2f5-9b4cf3cc912c
env = sqrt.(shot .* shot + hilb .* hilb);

# ╔═╡ c7899ab8-6d2d-49c3-85f5-40684c9e9dfc
begin
	pabs = heatmap(x, t, abs.(shot[1:1001,:]), 
	            yflip=true, legend=:false, clim=(0, 0.25),
	            xlabel="Offset (km)", ylabel="Time (s)", 
	            title="(b) Absolute Value", cmap=:grays)
	penv = heatmap(x, t, env[1:1001,:], 
	            yflip=true, legend=:false, clim=(0, 0.25),
	            xlabel="Offset (km)", ylabel="Time (s)", 
	            title="(c) Envelope", cmap=:grays)
	plot(pshot, pabs, penv, layout=(1, 3))
end

# ╔═╡ ca9c2845-e8b0-4a0f-bdf0-af496b490da3
md"""
## Local frequency

One practical local attribute is local frequency, which expands on the concept of instantaneous frequency.
"""

# ╔═╡ 50e47065-1e67-425e-a896-bbef1607915a
md"""
For a real-valued time-variable signal $f(t)$, the complex analytical trace is defined as

$$c(t) = f(t) + i\,h(t)\;,$$

where $h(t)$ is the Hilbert transform of $f(t)$. 

We can also represent the complex trace in terms of the envelope $A(t)$ and the instantaneous phase $\phi(t)$, as follows:

$$c(t) = A(t)\,e^{i\,\phi(t)}\;.$$

By definition, the instantaneous frequency is the time derivative of the instantaneous phase:

$$\omega(t) = \phi'(t) = \mathit{Im}\left[\frac{c'(t)}{c(t)}\right] = \frac{f(t)\,h'(t) - f'(t)\,h(t)}{f^2(t) + h^2(t)}\;.$$
"""

# ╔═╡ 0b16dcad-bb2c-4f06-83b5-e4173616ca92
md"""
!!! assignment

    ## Task 2 (theoretical)
    Derive the equation shown above.
"""

# ╔═╡ 0f233ad8-1450-42b1-b671-8a02a906f037
md"""
!!! assignment

    ## Task 3
    To implement local frequency, replace the point-by-point division in the definition of the instantaneous frequency with a smooth division. Write a function to calculate local frequency and apply it to the AGC-corrected shot gather from Task 1.
"""

# ╔═╡ 71f98c03-af3c-40da-8385-a223dd867ed3
md"""
## Local orthogonalization

When addressing the challenge of separating signal from noise, selecting the right parameters for a clear separation can be difficult. If we lean towards removing as much noise as possible and end up with some signal leaking into the estimated noise, we can improve the separation by matching the estimated signal $\mathbf{s}_0$ to the remaining signal in the separated noise $\mathbf{n}_0$.
"""

# ╔═╡ 4b5b4d31-a351-4c78-a2e2-212573c484f6
md"""
Doing it by simple scaling amounts to minimizing

$$\displaystyle \min_{\gamma} \left|\mathbf{n}_0 - \gamma\,\mathbf{s}_0\right|^2\;.$$
   
Estimating $\gamma$ by least-squares minimization, we can define the new signal $\mathbf{s}$ and the new noise $\mathbf{n}$ according to

$$\begin{array}{rcl}
  \mathbf{s}  & = & \mathbf{s}_0 + \gamma\,\mathbf{s}_0 = (1+\gamma)\,
                    \mathbf{s}_0\;, \\
  \mathbf{n}  & = & \mathbf{n}_0 - \gamma\,\mathbf{s}_0\;.
  \end{array}$$
"""

# ╔═╡ d476dcc3-9bec-4a99-ac45-443ff9bdd01a
md"""
Notice that

$$\mathbf{s}^T\mathbf{n} =
  (1+\gamma)\,\left(\mathbf{s}_0^T\mathbf{n}_0 - \gamma\,
    \mathbf{s}_0^T\,\mathbf{s}_0\right) = \displaystyle
   (1+\gamma)\,\left(\mathbf{s}_0^T\mathbf{n}_0 -
       \frac{\mathbf{s}_0^T\mathbf{n}_0}{\mathbf{s}_0^T\,\mathbf{s}_0}\,
         \mathbf{s}_0^T\,\mathbf{s}_0\right)  = 0\;.$$

In other words, the new signal and noise estimates will be orthogonal.

The method becomes significantly more powerful when replacing scaling by a constant with scaling by a variable vector $\mathbf{c}$, defined via smooth division.
"""

# ╔═╡ 6ffe65ea-1734-4bfe-9b4b-2e0fa95428e3
md"""
* Chen, Y., and S. Fomel, 2015, Random noise attenuation using local signal-and-noise orthogonalization: Geophysics, 80, WD1–WD9.
"""

# ╔═╡ 0e06320c-9fc3-4903-a7de-621d802336aa
md"""
### Random noise

We will first apply local orthogonalization for random noise removal. The data example is 2-D seismic reflection data from the National Petroleum Reserve in Alaska.

* [https://wiki.seg.org/wiki/Alaska\_2D\_land\_line\_31-81](https://wiki.seg.org/wiki/Alaska_2D_land_line_31-81 )
"""

# ╔═╡ c821912f-7fce-4ba9-88aa-3621395ebb55
begin
	url = "http://s3.amazonaws.com/open.source.geoscience/open_data/alaska/line31-81/31_81_PR.SGY"
	download(url, "alaska.sgy")
end

# ╔═╡ 6f4ae5d6-9bf9-4fcd-860b-b060998264eb
begin
	seismic = segy_read("alaska.sgy");
	alaska = Float32.(seismic.data[1:1001,:]);
end

# ╔═╡ 1b44f777-d2c1-45e6-b6da-bd15c3832134
nt, nx = size(alaska);

# ╔═╡ edce55b9-877c-4c92-a0c6-6553f48c63fe
ta = range(start=0, length=nt, step=0.004); # time axis

# ╔═╡ 76622b9d-1c78-4235-942e-8e863548e977
plot_alaska(data, title) = heatmap(1:nx, ta, data, clim=(-2000, 2000), 
	                               legend=:none, yflip=true, cmap=:grays, 
	                               title=title, ylabel="Time (s)")

# ╔═╡ b0128c91-3de1-4486-aec4-eb23b09f8a3c
plot_alaska(alaska, "Seismic Stack from Alaska")

# ╔═╡ 6cb60721-fe08-4afe-aa5d-bae37a77aaf6
mutable struct HelixFilter
    lag::Vector{CartesianIndex}
    flt::Vector
    HelixFilter(lag,flt) = new(map(CartesianIndex,lag),flt)
	HelixFilter(lag) = HelixFilter(lag,zeros(Float32, length(lag)))
end

# ╔═╡ 8df3d734-78ca-4c04-879d-76107ae80146
function helix(a::HelixFilter, ci::CartesianIndices)
    "convert helix lags to 1-D for a given grid"
    # middle of the grid
    mid = CartesianIndex(Tuple(last(ci)) .÷ 2)
    # helix index of middle
    hmid = LinearIndices(ci)[mid]
    # from Cartesian shift to helix shift
    return LinearIndices(ci)[map(x -> x + mid, a.lag)] .- hmid
end

# ╔═╡ 36fbec7a-691f-4c35-9b42-255039547cbd
Base.length(a::HelixFilter) = Base.length(a.lag)

# ╔═╡ c75e3970-ca4c-46fc-b4ee-313d2f795527
function deriv(x::Vector{T}; order=6) where T <: Real
    "derivative filter"
    n = length(x) # nt
    t, y = deepcopy(x), similar(x) # h, trace2
    for it in order:-1:1
        for i in 2:n-1
            y[i] = t[i]-(t[i+1]+t[i-1])/2
        end
        y[1], y[n] = y[2], y[n-1]
        for i in 1:n
            t[i] = x[i] + y[i]*it/(2*it+1)
        end
    end
    y[1] = t[2]-t[1]
    for i in 2:n-1
        y[i] = (t[i+1]-t[i-1])/2
    end
    y[n] = t[n]-t[n-1]
    return y
end

# ╔═╡ 7c4ddd84-62dc-49f9-854d-c0bf654dff04
function pef_lag2(n1, n2)
	# find PEF lags in 2D
	nc = floor(Int, n1//2)
	lag = Array{Tuple{Int, Int}}(undef, n1-nc-1 + n1*(n2-1) )
	i = 1
	for i2 in 1:n2
		n0 = i2 == 1 ? nc+2 : 1
		for i1 in n0:n1
			lag[i] = (i1-nc-1, i2-1)
			i += 1
		end
	end
	return lag
end

# ╔═╡ 3b1742c2-ed7e-415a-990f-974154d9e80b
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

# ╔═╡ 8c4e1ef3-45f3-4435-8332-c58e1feabc55
function hrecursive(x, a::HelixFilter, adjoint=false)
    cx, nx, na = CartesianIndices(x), length(x), length(a)
    lag = helix(a, cx)
    y = similar(x)
    if (adjoint)
        for ix in nx:-1:1
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
        for ix in 1:nx
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

# ╔═╡ 0bf31a7b-d4f0-41cd-a05b-f34ba4ab73b1
function pef(f, x, a::HelixFilter, mask)
	# convolution with a prediction-error filter
    nx, na = length(x), length(a)
    lag = helix(a, CartesianIndices(x))
	y = zeros(eltype(x), size(x))
    for ia in 1:na
        for iy in 1 + lag[ia] : nx
			if mask[iy] # skip missing points
	            ix = iy - lag[ia]    
                y[iy] -= x[ix] * f[ia]
            end
        end
    end
	return y
end

# ╔═╡ 8e5cc7a1-83a8-4f3f-bba1-3cbb80747af9
function pef_adjoint(y, x, a::HelixFilter, mask)
	# adjoint convolution with a prediction-error filter
    nx, na = length(x), length(a)
    lag = helix(a, CartesianIndices(x))
	f = zeros(eltype(a.flt), na)
    for ia in 1:na
        for iy in 1 + lag[ia] : nx
			if mask[iy] # skip missing points
	            ix = iy - lag[ia]
				f[ia] -= x[ix] * y[iy] 
			end
        end
    end
	return f
end

# ╔═╡ 20347efe-57c1-4d93-ab16-53066e4e34d3
function conjgrad(forward::Function, adjoint::Function, d::Array, x0::Array, niter::Int)
    "Conjugate-gradients for minimizing |forward(x) - d|^2"
    x = deepcopy(x0)
    R = forward(x) .- d  
    s, S = similar(x), similar(d)
    gnp = zero(eltype(x))
    for iter in 1:niter
        g = adjoint(R)
        G = forward(g)
        gn = g ⋅ g
        @show iter, gn     
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
    end
    return x
end

# ╔═╡ 1b9f25af-3a03-46c2-b116-01b886bc3c13
# conjugate-gradients with preconditioning
function conjgrad(forward::Function, adjoint::Function, 
	              precon::Function, precon_adjoint::Function,
                  d::Array, x0::Array, ϵ::Real, niter::Int)
    "Conjugate-gradients for minimizing |forward(x)-d|^2 + ϵ^2*|regul(x)|^2"
    x = deepcopy(x0)
    R1 = forward(precon(x)) - d
    R2 = ϵ*x
    nd = length(d)
    s, S1, S2 = similar(x), similar(R1), similar(R2)
    gnp = zero(eltype(x))
    for iter in 1:niter
        g = precon_adjoint(adjoint(R1)) + ϵ*R2 # block adjoint
        G1 = forward(precon(g))
        G2 = ϵ*g
        gn = g ⋅ g   
        if iter==1 # steepest descent
            s, S1, S2 = g, G1, G2
        else            
            β = gn/gnp
            s = g + β*s
            S1 = G1 + β*S1
            S2 = G2 + β*S2
        end
        gnp = gn      
        α = -gn/(S1 ⋅ S1 + S2 ⋅ S2)
        x += α*s
        R1 += α*S1
        R2 += α*S2
		Rn = real(R1 ⋅ R1 + R2 ⋅ R2) # length of the residual
		@show iter, Rn
    end
    return precon(x)
end

# ╔═╡ 1f6a8278-a66d-40f2-8b01-7ad1e501a57b
function smooth_division(numerator::Array, denominator::Array, 
	                     radius::Vector{Int}; niter=100)
    n = length(numerator)
    p0 = similar(numerator)
    fill!(p0, zero(eltype(p0)))
    # normalization
    norm = denominator ⋅ denominator
    if norm == 0.0
        return p0
    end
    norm = sqrt(n/norm)
    # weighting function
    weight = x -> x .* denominator .* norm
    return conjgrad(weight, weight, 
		            x -> LocalSignalAttributes.smooth(x, radius), 
					1.0, numerator * norm, p0, niter)
end 

# ╔═╡ 860998b1-d3cd-44bb-b9b4-08924919ec4e
function agc(input::Array, radius::Vector{Int}; niter=100)
	# automatic gain control
    ione = ones(size(input))
    gain = abs.(input)
    return smooth_division(ione, gain, radius, niter=niter)
end

# ╔═╡ 5593902a-8988-4047-b37e-a135a992450d
gain = agc(shot, [10, 1000], niter=500);

# ╔═╡ 04cdf217-96bd-4700-aa9b-08f1092ae83d
gained = gain .* shot;

# ╔═╡ 74b5b171-ee33-4c60-9bcc-7dfb0a52785f
begin
	pg = heatmap(x, t, gain[1:1001,:], 
	            yflip=true, legend=:false,
	            xlabel="Offset (km)", ylabel="Time (s)", 
	            title="(b) Gain")
	pa = plot_seismic(gained, "(c) AGC")
	plot(pshot, pg, pa, layout=(1, 3))
end

# ╔═╡ 2efdb093-b329-4172-ae7a-d7de252cdb5f
function find_mask(known, lag) 
	# make a filter of all ones
	a = HelixFilter(lag)
	a.flt .= one(Float32)

	# sert ones at unknown locations
	x = ones(Float32, size(known))
	x[known] .= zero(Float32)

	y = hconvolve(x, a)

	return y .== zero(Float32)
end

# ╔═╡ 9af9198a-4d9a-42b9-9f2f-6e71f55532ff
nlag = pef_lag2(3, 1)

# ╔═╡ 28029ffb-0734-42b1-9bc0-7f3199a62087
slag = pef_lag2(5, 3)

# ╔═╡ ac87023b-5a32-4983-80c1-5d3d8563a7bf
function find_pef(data, lag; niter=length(lag))
	f = zeros(eltype(data), length(lag)) 
	a = HelixFilter(lag, f)
	mask = find_mask(ones(Bool,size(data)), lag)
	a.flt = conjgrad(f -> pef(f, data, a, mask), 
		             x -> pef_adjoint(x, data, a, mask),  
					 data, f, niter)
	return a
end

# ╔═╡ 25dadfb9-e238-4dec-bc8c-b4d427b1d2d8
function signal_noise(data, signal_pef, noise_pef; ϵ=1.0, niter=50)
	# PEF-based signal and noise separation
	forward(x) = hconvolve(x, noise_pef, false)
	adjoint(x) = hconvolve(x, noise_pef, true)
	precon(x) = hrecursive(x, signal_pef, false)
	precon_adjoint(x) = hrecursive(x, signal_pef, true)
	x0 = zeros(eltype(data), size(data))

	signal = conjgrad(forward, adjoint, precon, precon_adjoint,
                      forward(data), x0, ϵ, niter)
	return signal, data-signal
end

# ╔═╡ 0eddace1-096c-4f02-b7d3-67cea1a9997f
begin
	npef = find_pef(alaska, nlag, niter=4)
	spef = find_pef(alaska, slag, niter=30)
end

# ╔═╡ ba736e6e-698d-4ef4-99f8-cef3dd51538f
as0, an0 = signal_noise(alaska, spef, npef, ϵ=4);

# ╔═╡ 45b909fa-f2cc-4706-bc9b-a6544b44245b
plot_alaska(as0, "Estimated Signal")

# ╔═╡ 5b03f6fc-e7d2-45bd-b631-0ea60b770120
plot_alaska(an0, "Estimated Noise")

# ╔═╡ afa29dc8-09fc-4448-807b-5a486745cb62
ns = smooth_division(an0, as0, [100, 10], niter=200);

# ╔═╡ 5157058c-f80e-41e6-a7cf-a313eb470bd6
begin
	an1 = an0 - ns .* as0
	as1 = alaska - an1
end

# ╔═╡ a00a5caa-8788-41a9-91ef-3928dc76856f
plot_alaska(as1, "Improved Signal")

# ╔═╡ 498919cd-3d4a-443d-b354-fb16a90246a2
plot_alaska(an1, "Improved Noise")

# ╔═╡ aa3911f6-5a69-40c8-9994-649affe8958e
md"""
### Coherent noise

For a case of coherent noise, our initial denoising will involve bandpass filtering.
"""

# ╔═╡ 1d51c27f-23e6-4058-aaee-a9f153656838
spectrum = mean(abs.(rfft(gained, 1)), dims=2);

# ╔═╡ f3ce450a-d08b-4f5c-8b36-43df1f0321cd
fs = rfftfreq(2000, 1/0.002); # frequency axis

# ╔═╡ 17866ac6-0b9d-4ce5-aec5-d771b3a23a68
plot(fs, spectrum, title="Spectrum", xlabel="Frequency (Hz)", 
	 linewidth=2, label=:none)

# ╔═╡ 932c0b5a-695a-4637-9708-068e2f51554f
plot(fs, spectrum, title="Spectrum at Low Frequencies", xlabel="Frequency (Hz)",
     linewidth=2, label=:none, xlim=[0,50])

# ╔═╡ 38a63208-3d0b-4aa5-8cee-02f7f318f242
function butterworth!(lowpass::Bool, x::Vector{T}, 
        freq, np::Int) where T <: Real
    "Bandpass filtering using Butterworth filters"
    nx = length(x)
    arg = 2π * freq
    sinf, cosf = sin(arg), cos(arg)
    fact = lowpass ? sin(arg/2) : cos(arg/2)
    fact *= fact 
    for j in 1:np # loop over poles
        ss = sin(π*(2*j-1)/(4*np))*sinf 
        # denominator filter
        d0, d1, d2 = fact/(1+ss), -2*cosf/fact, (1-ss)/fact
        # forward and backward convolution
        for range in (1:nx, nx:-1:1)
            x1 = x0 = y1 = y2 = zero(T)
            for ix in range
                x2, x1, x0 = x1, x0, x[ix]
                y0 = lowpass ? (x0 + 2*x1 + x2 - d1 * y1 - d2 * y2)*d0 :
                               (x0 - 2*x1 + x2 - d1 * y1 - d2 * y2)*d0
                y2, y1, x[ix] = y1, y0, y0
            end
        end
    end
end

# ╔═╡ 4a586817-91fc-4879-b86b-dd5ace2dfe82
function bandpass(x::Vector{T}, f1, f2, np::Int, dt) where T <: Real
    y = deepcopy(x)
    if f1 != :none # high-pass filtering
        butterworth!(false, y, f1*dt, np)
    end
    if f2 != :none # low-pass filtering
		butterworth!(true, y, f2*dt, np)
	end
    return y
end

# ╔═╡ ef6d2f28-5925-4106-b39c-7e19f56976fd
noise0 = mapslices(trace -> bandpass(trace, :none, 25, 10, 0.002), 
	               gained; dims=1);

# ╔═╡ e1654134-c155-443c-8cdf-e36da05f990c
signal0 = gained - noise0;

# ╔═╡ e88e8b33-b696-4bb3-92da-d733c0fbf074
begin
	pd = plot_seismic(gained, "(a) Data")
	pn0 = plot_seismic(noise0, "(b) Est. Noise")
	ps0 = plot_seismic(signal0, "(c) Est. Signal")
	plot(pd, pn0, ps0, layout=(1, 3))
end

# ╔═╡ 0c5d273a-9998-4119-8b3e-af287b49aeb8
γ = smooth_division(noise0, signal0, [10, 2], niter=200);

# ╔═╡ 97ce925f-f3ad-4191-835d-a2eb745d63b3
begin
	noise = noise0 - γ .* signal0
	signal = gained - noise
end

# ╔═╡ e2ce2606-4433-4129-a7c9-2b635dd67bcd
begin
	pn = plot_seismic(noise, "(b) Est. Noise")
	ps = plot_seismic(signal, "(c) Est. Signal")
	plot(pd, pn, ps, layout=(1, 3))
end

# ╔═╡ 6f445b0c-e0d7-4f44-b403-970b0f328046
md"""
!!! assignment

    ## Task 4

    Repeat the exercise above, but switch the roles of noise and signal. 
    1. Use a different cut-off frequency `f2` in the call to `bandpass` so that the initial estimated noise is signal-free while the initial signal is contaminated with noise.
	2. Orthogonalize signal and noise through minimization
    $$\displaystyle \min_{\gamma} \left|\mathbf{s}_0 - \gamma\,\mathbf{n}_0\right|^2\;.$$
    3. Compare the results.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LocalSignalAttributes = "0a92bf9b-4da3-44e8-9286-830175b27cf8"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUIExtra = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"
SegyIO = "157a0f19-4d44-4de5-a0d0-07e2f0ac4dfa"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
FFTW = "~1.8.1"
HTTP = "~1.10.15"
LocalSignalAttributes = "~1.0.3"
Plots = "~1.40.10"
PlutoUIExtra = "~0.1.8"
SegyIO = "~0.8.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "d7716c5c7e597638fb2fdec368c57a95863f1984"

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
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

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
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

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

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

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
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

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

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "7de7c78d681078f027389e067864a8d53bd7c3c9"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+3"

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
git-tree-sha1 = "0ff136326605f8e06e9bcf085a356ab312eef18a"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.13"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "9cb62849057df859575fc1dda1e91b82f8609709"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.13+0"

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

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

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
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

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
git-tree-sha1 = "cd714447457c660382fe634710fb56eb255ee42e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.6"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

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
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

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

[[deps.LocalSignalAttributes]]
deps = ["FFTW", "LinearAlgebra"]
git-tree-sha1 = "d549500d102c67cf5240b219f4844c0972e36ebc"
uuid = "0a92bf9b-4da3-44e8-9286-830175b27cf8"
version = "1.0.3"

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

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

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
git-tree-sha1 = "cc0a5deefdb12ab3a096f00a6d42133af4560d71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.2"

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
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3b31172c032a1def20c98dae3f2cdc9d10e3b561"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.1+0"

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
git-tree-sha1 = "564b477ae5fbfb3e23e63fc337d5f4e65e039ca4"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.10"

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
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

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
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SegyIO]]
deps = ["Distributed", "Printf", "Test"]
git-tree-sha1 = "0fc24db28695a80aa59c179372b11165d371188a"
uuid = "157a0f19-4d44-4de5-a0d0-07e2f0ac4dfa"
version = "0.8.5"

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
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56c6604ec8b2d82cc4cfe01aa03b00426aac7e1f"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.4+1"

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
git-tree-sha1 = "068dfe202b0a05b8332f1e8e6b4080684b9c7700"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.47+0"

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

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

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
# ╟─1b9d9a5e-fef4-41d3-aede-320bbbd2fd4b
# ╟─7edeb330-aa60-4757-9fcd-57ce154281e1
# ╟─8d4c7897-54ce-4d98-b3d9-eeff291ebdcc
# ╟─48e11ea3-d612-4822-b849-cba6b86f6fc5
# ╟─2aecc41f-6276-436b-91fb-65137ddcef54
# ╟─48009579-80cd-44a8-ba01-62d847f91a82
# ╟─4f346e49-8555-4629-a4e9-a1c10ec91636
# ╟─25c51cfe-fe0f-468a-9f9d-e99a39aca7c6
# ╠═d86563fd-98e3-44d0-be77-071a94241f5e
# ╠═ff4dfd2f-fc26-411d-9968-ce8541d9f082
# ╠═43495c00-e48c-43e7-9281-743342b2597e
# ╠═2f845a97-486e-4daa-a88b-a784ee4e52eb
# ╠═6da8c15c-cb31-4a7e-849d-40e48cd1630e
# ╠═58c1b007-7fa2-470a-a82f-f968ec48ff3f
# ╠═fc5b3ccd-8aed-4930-ab9d-5d8b057d72ba
# ╠═3a80d65d-b1ff-4366-b058-cb1096d5574d
# ╠═6486abf2-97ea-4a8d-b98a-1ea4ed9168db
# ╠═c24ed93d-fd1c-41dd-8fdf-e542bd8b0ea2
# ╠═1f6a8278-a66d-40f2-8b01-7ad1e501a57b
# ╠═860998b1-d3cd-44bb-b9b4-08924919ec4e
# ╠═5593902a-8988-4047-b37e-a135a992450d
# ╠═04cdf217-96bd-4700-aa9b-08f1092ae83d
# ╠═74b5b171-ee33-4c60-9bcc-7dfb0a52785f
# ╟─2dfa5cf8-198a-497c-8542-fe5e432df176
# ╠═53401cc8-9710-4d00-bdff-ec0e3655165e
# ╠═2117d334-20e7-45ca-b2f5-9b4cf3cc912c
# ╠═c7899ab8-6d2d-49c3-85f5-40684c9e9dfc
# ╟─ca9c2845-e8b0-4a0f-bdf0-af496b490da3
# ╟─50e47065-1e67-425e-a896-bbef1607915a
# ╟─0b16dcad-bb2c-4f06-83b5-e4173616ca92
# ╟─0f233ad8-1450-42b1-b671-8a02a906f037
# ╠═c75e3970-ca4c-46fc-b4ee-313d2f795527
# ╟─71f98c03-af3c-40da-8385-a223dd867ed3
# ╟─4b5b4d31-a351-4c78-a2e2-212573c484f6
# ╟─d476dcc3-9bec-4a99-ac45-443ff9bdd01a
# ╟─6ffe65ea-1734-4bfe-9b4b-2e0fa95428e3
# ╟─0e06320c-9fc3-4903-a7de-621d802336aa
# ╠═c821912f-7fce-4ba9-88aa-3621395ebb55
# ╠═77fb049f-f800-4ccc-a662-b0ab59cc3a2b
# ╠═6f4ae5d6-9bf9-4fcd-860b-b060998264eb
# ╠═1b44f777-d2c1-45e6-b6da-bd15c3832134
# ╠═edce55b9-877c-4c92-a0c6-6553f48c63fe
# ╠═76622b9d-1c78-4235-942e-8e863548e977
# ╠═b0128c91-3de1-4486-aec4-eb23b09f8a3c
# ╟─6cb60721-fe08-4afe-aa5d-bae37a77aaf6
# ╟─8df3d734-78ca-4c04-879d-76107ae80146
# ╟─36fbec7a-691f-4c35-9b42-255039547cbd
# ╟─7c4ddd84-62dc-49f9-854d-c0bf654dff04
# ╟─3b1742c2-ed7e-415a-990f-974154d9e80b
# ╟─8c4e1ef3-45f3-4435-8332-c58e1feabc55
# ╟─0bf31a7b-d4f0-41cd-a05b-f34ba4ab73b1
# ╟─8e5cc7a1-83a8-4f3f-bba1-3cbb80747af9
# ╟─20347efe-57c1-4d93-ab16-53066e4e34d3
# ╟─1b9f25af-3a03-46c2-b116-01b886bc3c13
# ╟─2efdb093-b329-4172-ae7a-d7de252cdb5f
# ╠═9af9198a-4d9a-42b9-9f2f-6e71f55532ff
# ╠═28029ffb-0734-42b1-9bc0-7f3199a62087
# ╠═ac87023b-5a32-4983-80c1-5d3d8563a7bf
# ╠═25dadfb9-e238-4dec-bc8c-b4d427b1d2d8
# ╠═0eddace1-096c-4f02-b7d3-67cea1a9997f
# ╠═ba736e6e-698d-4ef4-99f8-cef3dd51538f
# ╠═45b909fa-f2cc-4706-bc9b-a6544b44245b
# ╠═5b03f6fc-e7d2-45bd-b631-0ea60b770120
# ╠═afa29dc8-09fc-4448-807b-5a486745cb62
# ╠═5157058c-f80e-41e6-a7cf-a313eb470bd6
# ╠═a00a5caa-8788-41a9-91ef-3928dc76856f
# ╠═498919cd-3d4a-443d-b354-fb16a90246a2
# ╟─aa3911f6-5a69-40c8-9994-649affe8958e
# ╠═ad03ff21-1a42-45af-8036-54a22efe9bcb
# ╠═1d51c27f-23e6-4058-aaee-a9f153656838
# ╠═f3ce450a-d08b-4f5c-8b36-43df1f0321cd
# ╠═17866ac6-0b9d-4ce5-aec5-d771b3a23a68
# ╠═932c0b5a-695a-4637-9708-068e2f51554f
# ╠═38a63208-3d0b-4aa5-8cee-02f7f318f242
# ╠═4a586817-91fc-4879-b86b-dd5ace2dfe82
# ╠═ef6d2f28-5925-4106-b39c-7e19f56976fd
# ╠═e1654134-c155-443c-8cdf-e36da05f990c
# ╠═e88e8b33-b696-4bb3-92da-d733c0fbf074
# ╠═0c5d273a-9998-4119-8b3e-af287b49aeb8
# ╠═97ce925f-f3ad-4191-835d-a2eb745d63b3
# ╠═e2ce2606-4433-4129-a7c9-2b635dd67bcd
# ╟─6f445b0c-e0d7-4f44-b403-970b0f328046
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
