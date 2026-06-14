### A Pluto.jl notebook ###
# v0.20.27

#> [frontmatter]
#> chapter = 4
#> section = 17
#> order = 17
#> image = "https://raw.githubusercontent.com/fonsp/Pluto.jl/580ab811f13d565cc81ebfa70ed36c84b125f55d/demo/plutodemo.gif"
#> title = "Sparsity"
#> tags = ["module4", "track_julia", "track_material", "Pluto", "PlutoUI"]
#> layout = "layout.jlhtml"

using Markdown
using InteractiveUtils

# ╔═╡ 668405bb-7ade-4455-8697-34b1747f0b26
begin
	using PlutoUIExtra
	TableOfContents()
end

# ╔═╡ 748715c1-5a28-4b07-ba48-70b99880f9cd
using Plots

# ╔═╡ 09851753-bfa7-4c7c-9502-0937835d9240
using FFTW

# ╔═╡ a8c11790-7e42-4880-99c8-ce349d6deec6
using LinearAlgebra, Random

# ╔═╡ ef693662-cfa4-4dd6-9723-8d6485d03f6e
using DelimitedFiles

# ╔═╡ 25684269-c636-46a0-88fa-1bf71fa457a4
using LaTeXStrings

# ╔═╡ b5118e1d-5263-47c8-8e91-e90f9edaafae
using Statistics

# ╔═╡ cb9e8b2b-8346-4696-a3c3-d335ec246afc
import HTTP

# ╔═╡ 6632dc0e-b2c9-4294-96a3-e008d3b98d01
let
    notebook_dir = @__DIR__
    toc_path = joinpath(notebook_dir, "TOC_Notebook.jl")
    
    Markdown.parse("[⬅️ Back to Table of Contents](./open?path=$(HTTP.escapeuri(toc_path)))")
end

# ╔═╡ 14a0572a-f427-4103-ae9d-06fd0ce2ef0f
md"""
# Sparsity

The principle of sparsity plays a vital role in data analysis. Finding a sparse representation of signals in a transform domain is a convenient way to characterize signal patterns. This chapter examines methods for enforcing sparsity through regularized inversion and explains how to interpret sparsity constraints from both shaping and optimization perspectives.
"""

# ╔═╡ b0da32b2-aa83-4f9c-9456-1caa60baeab3
md"""
## Shaping for sparsity

A sparse model has only a few non-zero values. A simple transformation that makes an input model sparse is *thresholding*, which sets small values to zero. *Hard thresholding* does exactly that and can be defined mathematically as a point-by-point operation

$$h_{\tau}[x_n] = \left\{ \begin{array}{rl} x_n & \mbox{if} \quad |x_n| \ge \tau\;, \\ 0 & \mbox{if} \quad |x_n| < \tau\;. \end{array}\right.$$ 

Here, $\tau$ represents the threshold value. *Soft thresholding*, also known as *shrinkage*, further "shrinks" larger coefficients toward zero, as follows:

$$s_{\tau}[x_n] = \left\{ \begin{array}{rl} x_n-\tau & \mbox{if} \quad x_n \ge \tau\;, \\ 0 & \mbox{if} \quad |x_n| < \tau\;, \\ x_n+\tau & \mbox{if} \quad x_n \le -\tau\;. \end{array}\right.$$
"""

# ╔═╡ d7bb2b15-413e-4631-9de5-a961d7103611
md"""
We can devise other forms of a point-by-point sparsifying transformation based on a threshold. For example, we can multiply the input by a nonlinear weight

$$w_{\tau}[x_n] = x_n\,e^{-\tau^2/x_n^2}\;.$$

If $|x_n|$ is much larger than $\tau$, $w_{\tau}[x_n]$ approaches $x_n$. On the other hand, if $|x_n|$ is much smaller than $\tau$, $w_{\tau}[x_n]$ approaches zero.
"""

# ╔═╡ 9bd5832e-67a2-4ccf-9a39-712ebfc46cdb
begin
	hard(x, τ=3) = (abs(x) > τ) ? x : zero(x)
	soft(x, τ=3) = (abs(x) > τ) ? x - sign(x) * τ : zero(x)
	thre(x, τ=3) = x * exp(-τ^2/x^2)
end

# ╔═╡ 92108c6b-948f-4cba-9ba1-d00d89a24094
plot([hard soft thre], -15, 15, 
	 labels=["hard" "soft" "exponential"], 
     title="Thresholding Functions", linewidth=2)

# ╔═╡ 3b15ba8c-1d8e-4660-93ec-6ed1afc1fdce
md"""
### Compressive sampling

What makes thresholding a useful operation? 

Classical sampling theory states that a band-limited signal can be exactly reconstructed if the sampling interval is less than the reciprocal of the maximum frequency. Let us consider a case where this condition is violated.
"""

# ╔═╡ c80d2f03-2bb1-4ba3-86e6-b3365a9e0a73
x = range(start=0,step=0.001,length=10000);

# ╔═╡ 24822111-42f6-4f14-9cb4-1344bee1de86
# example signal with two sinusoids
signal = @. 50*sin(20π * x + π^2/5) + 100*sin(80π * x + π^2/9);

# ╔═╡ d4545fc9-16f1-41c6-8cf9-1597eb3f38ea
# subsample
signal10 = signal[1:10:end];

# ╔═╡ 743bf67f-a2fd-46ee-8a43-e28dd3f43c7e
begin
	# select random values
	mask = rand(Float64, length(signal10))
	known = mask .> 0.75
	unknown = .!known
	signal10[unknown] .= 0;
end

# ╔═╡ bb48142b-4daf-4afe-8e74-5bd184f794a1
# Fourier transform (real to complex)
fsignal10 = rfft(signal10);

# ╔═╡ fbb2d10f-2001-446c-a108-456330a98812
# interpolate with padding in the Fourier domain
interp = irfft(vcat(fsignal10, zeros(5001-501)), 10000) * 10;

# ╔═╡ 075920c6-5496-4ebe-a77e-fca3fe205d78
begin
	plot(x, [signal interp], xlim=[0, 0.5], ylim=[-160, 160],
		 linewidth=2, 
	     labels=["ideal" "randomly subsampled and interpolated"])
	scatter!(x[1:10:end][known], signal10[known], label=:none, color=:lightblue)
end

# ╔═╡ 3f293e34-6b6e-4841-8543-84ae12fe4ed5
# frequencies
f = rfftfreq(length(signal10));

# ╔═╡ e58e1fa1-d0f5-4151-a037-5b42dfc10e55
plot(f, abs.(fsignal10), linewidth=2, title="Spectrum", 
     xlabel="frequency", color=:green, label=:none)

# ╔═╡ 09541de5-56ba-4050-90b7-75a11edba450
md"""
We can see that the signal is sparse in the Fourier domain, with only two significant coefficients. To reconstruct the signal, we can apply thresholding.
"""

# ╔═╡ a4ef85a2-7a41-44a2-9541-ebb65758f99e
threshold = hard.(fsignal10, 4000);

# ╔═╡ 9594ef17-371c-4803-9f39-5ae94e9e879e
plot(f, abs.(threshold), linewidth=2, 
	 title="Spectrum After Thresholding", 
     xlabel="frequency", color=:green, label=:none)

# ╔═╡ c51884ae-90af-4fe0-83c9-aee5a8399d5b
tinterp = irfft(vcat(threshold, zeros(5001-501)), 10000) * 40;

# ╔═╡ 451655cb-2926-44ca-aab1-e12a2bf21f17
plot(x, [signal tinterp], xlim=[0, 0.5], ylim=[-160, 160],
	 linewidth=2, labels=["ideal" "reconstruction"])

# ╔═╡ 7b205fa2-af01-477f-9aec-a392965b998c
md"""
The principle illustrated here is known as *compressive sensing* or *compressive sampling* and refers to the ability to reconstruct data from random samples when the transform domain is sparse.
"""

# ╔═╡ f4927c25-652c-49bb-af20-bb690ddf3dd8
md"""
* Bougher, B., 2015, Introduction to compressed sensing: The Leading Edge, 34, 1256–1257.
"""

# ╔═╡ d16aebc9-209f-43de-8aaf-e0f5dd7dbe6d
md"""
## Connection with optimization

Suppose that, given an input data vector $\mathbf{y}$, we look for the
vector $\mathbf{x}$ by minimizing the following least-squares
objective function:

$$\|\mathbf{x}-\mathbf{y}\|^2 + \epsilon^2\,\|\mathbf{x}\|^2 = \sum_{n=1}^{N} \left[(x_n-y_n)^2 + \epsilon^2\,x_n^2\right]$$

for a fixed scalar $\epsilon$. By taking the component-by-component
derivative in $x_n$, we can find that the objection function is minimized by solving the equation

$$\mathbf{x} - \mathbf{y} + \epsilon\,\mathbf{x} = 0\;,$$

which leads to

$$\mathbf{x} = \frac{1}{1+\epsilon^2}\,\mathbf{y}\;.$$

In other words, the solution corresponding to the $L_2$ regularization is to make $\mathbf{x}$ a bit smaller than $\mathbf{y}$.
"""

# ╔═╡ 15b094f1-2f1c-48b4-a78e-65428c3e882a
md"""
What if we change the second term in the objective function to the $L_1$-norm? The objective function becomes

$$\|\mathbf{x}-\mathbf{y}\|_2^2 + 2\,\mu\,\|\mathbf{x}\|_1 = \sum_{n=1}^{N} \left[(x_n-y_n)^2 + 2\,\mu\,|x_n|\right]$$

for some constant $\mu$. To minimize the $L_1$ objective function, we can set to zero derivatives with respect to $x_n$, obtaining

$$x_n - y_n + \mu\,\mbox{sign}\,{x_n} = 0\;.$$

To satisfy this equation, we can consider two cases:
1. $x_n > 0$ The equation becomes $x_n - y_n + \mu = 0$ and is satisfied by $x_n = y_n - \mu$ provided that $y_n > \mu$.
2. $x_n < 0$ The equation becomes $x_n - y_n - \mu = 0$ and is satisfied by $x_n = y_n + \mu$ provided that $y_n < -\mu$.

It is easy to see that the operation applied to the input $y_n$ is simply soft thresholding: $x_n = s_{\mu}[y_n]$.

As pointed out in the previous discussion of the difference between the mean and the median, changing the norm from $L_2$ to $L_1$ corresponds to changing the assumed probability distribution from normal (Gaussian) to exponential, thus allowing for a higher probability of outliers.
"""

# ╔═╡ 00c547fd-36dc-403d-be66-905259734ac1
md"""
Sometimes, using a norm with a hybrid behavior, which behaves like $L_2$ near minimum and like $L_1$ at the flanks,  makes sense in applications. For example, we can replace the absolute-value function $|x_n|$ with the hyperbolic function $\sqrt{|x_n|^2 + \epsilon^2}$ or with the *Huber norm*

$$H(x_n) =  \left\{  \begin{array}{rl}
 x_n^2                   & \mbox{if} \quad |x_n| \le \epsilon\;, \\
                \epsilon\,(2\,|x_n| - \epsilon), & \mbox{if} \quad |x_n| > \epsilon\;. \end{array}\right.$$ 

The $L_0$-norm refers to the number of non-zero elements. In this case, the objective function is

$$\|\mathbf{x}-\mathbf{y}\|_2^2 + \mu^2\,\|\mathbf{x}\|_0 = \sum_{n=1}^{N} \left[(x_n-y_n)^2 + \mu^2\,\delta_1(x_n)\right]\;,$$

where 

$$\delta_1(x) = \left\{  \begin{array}{rl} 1 & \mbox{if} \quad x \ne 0\;, \\ 0 & \mbox{if} \quad x = 0\;. \end{array}\right.$$

When $x_n=0$, the minimum of $\left[(x_n-y_n)^2 +
  \mu^2\,\delta_1(x_n)\right]$ is $y_n^2$. When $x_n  \ne 0$, the minimum should be larger: we can set it to $\mu^2$ by setting $x_n=y_n$. Thus, we uncover the operation

$$x_n = \left\{ \begin{array}{rl} y_n & \mbox{if} \quad |y_n| \ge \mu\;, \\ 0 & \mbox{if} \quad |y_n| < \mu\;. \end{array}\right.\;,$$

which is equivalent to hard thresholding: $x_n = h_{\mu}[y_n]$. The danger of using the $L_0$ norm in optimization is that the corresponding objective function is not convex.
"""

# ╔═╡ a1056c94-49d2-40db-9926-c15fa90b6357
md"""
* Candès, E. J., J. K. Romberg, and T. Tao, 2006, Stable signal recovery from incomplete and inaccurate measurements: Communications on Pure and Applied Mathematics, 59, 1207–1223.
* Candès, E. J., and M. B. Wakin, 2008, An introduction to compressive sampling: IEEE Signal Processing Magazine, 25, 21–30.
* Claerbout, J. F., and F. Muir, 1973, Robust modeling with erratic data: Geophysics, 38, 826–844.
* Huber, P. J., 2009, Robust statistics, 2nd ed.: Wiley, New York.
"""

# ╔═╡ a5be1a5b-0382-4e21-9ed5-d5d33a95b5bb
md"""
!!! assignment
    ## Task 1 (theoretical)

    Derive and plot the thresholding operator that corresponds to the Huber norm

    $$\|\mathbf{x}-\mathbf{y}\|_2^2 + \mu^2\,\|\mathbf{x}\|_H = \sum_{n=1}^{N} \left[(x_n-y_n)^2 + \mu^2\,H(x_n)\right]\;.$$
"""

# ╔═╡ 2a3c0498-f714-4dad-b6d7-d1a049d580d3
md"""
## Iterative thresholding

Following the framework of shaping regularization, we can write an iteration of model shaping as

$$\mathbf{m}_{n+1} = \mathbf{S}_m\left[\mathbf{m}_{n} + \mathbf{B\,d} - \mathbf{B\,F\,m}_n\right]\;,$$

where model shaping $\mathbf{S}_m$ corresponds to one of the sparsifying thresholding operations. In the particular case, when the backward operator $\mathbf{B}$ is the adjoint of the linear forward operator $\mathbf{F}$ ($\mathbf{B} = \mathbf{F}^T$) and $\mathbf{S}_m$ corresponds to soft thresholding, the iteration above is known as *iterative thresholding algorithm*, which converges to the minimum of

$$\|\mathbf{F\,m-d}\|_2^2 + 2\,\mu\,\|\mathbf{m}\|_1\;.$$

In this case, the forward operator $\mathbf{F}$ may include a transform (such as the Fourier transform) from the domain where we expect the model to be sparse.
"""

# ╔═╡ 38af1498-ce38-4ea9-baec-51dfc3c78e45
md"""
Let us solve our toy data reconstruction problem using iterative thresholding.
"""

# ╔═╡ e8365393-de2c-4a47-8ee3-b7d22e341033
# mask for the full signal
begin
	unknown2 = zeros(Bool, length(signal))
	for x1 in x[1:10:end][known]*1000
		unknown2[round(Int,x1+1)] = true
	end
end

# ╔═╡ e289b562-c16b-45e4-9ed0-47c90adf3371
# forward operator
function forw(x::Array)
	pad = ifft(x)
	pad[unknown2] .= zero(eltype(x))
	return pad
end

# ╔═╡ 062fd5cb-42ae-4845-93cd-d988a071d154
# backward operator
function back(y::Array)
	pad = deepcopy(y)
	pad[unknown2] .= zero(eltype(y))
	return fft(pad)/length(pad)
end

# ╔═╡ 8f92b618-b37f-4061-82a9-37639598dbeb
begin
	cdata = complex.(signal)
	cdata[unknown2] .= 0
end

# ╔═╡ cc83782b-2d84-4db6-b9d9-1a08cbe4a066
# dot-product test 
function dottest(forward::Function, adjoint::Function, 
                 m::Array, d::Array)
    "Dot-product test"
    mod = similar(m); rand!(mod)
    dat = similar(d); rand!(dat)
    println(" L[m]⋅d = $(forward(mod) ⋅ dat)")
    println("L'[d]⋅m = $(adjoint(dat) ⋅ mod)")
end

# ╔═╡ 1cb1a5a4-d3aa-41b2-9ced-82063d3790cf
function shape(forward::Function, backward::Function, 
	           shaping::Function, λ, x0::Array, niter::Int)
	x = deepcopy(x0)
	for iter in 1:niter
		x = shaping(x0 + x - λ*backward(forward(x)))
		@show iter, x ⋅ x
	end
	return x
end

# ╔═╡ a2fca1ec-2a6a-49ad-9284-aea3811d0478
dottest(forw, back, cdata, cdata)

# ╔═╡ 51cdb012-ff83-40fe-a142-aeed7d57f703
x0 = back(cdata);

# ╔═╡ 74ae169c-d1be-4972-a0be-d3ac58d81541
τ = 1 # threshold value

# ╔═╡ fb2a7ac8-a6b6-4015-8ec3-01f43ff0f36c
xhat = shape(forw, back, x -> soft.(x, τ), length(x0), x0, 5);

# ╔═╡ 302e9c8b-f286-494a-aaec-9f1be0afc435
plot(x, [signal real(ifft(xhat))*length(xhat)], xlim=[0, 0.5], ylim=[-160, 160],
	 linewidth=2, labels=["ideal" "iterative thresholding"])

# ╔═╡ b7095ce7-1d91-477b-9b8f-82f56da5176d
md"""
!!! assignment

    ## Task 2

    Investigate how the convergence of iterative thresholding depends on the threshold value $\tau$.
"""

# ╔═╡ 7797bff3-94a4-4635-b7f6-ad959f26560c
md"""
## Iteratively reweighted least squares

An alternative to iterative thresholding is a linearization approach, where the $L_1$ norm behavior is approximated by weighting. This approach is known as *iteratively reweighted least squares* or IRLS and can be effective in practice.

IRLS amounts to the minimization of

$$\|\mathbf{F\,m}_k-\mathbf{d}\|_2^2 + 2\,\mu\,\|\mathbf{W}_k^{-1}\,\mathbf{m}_k\|_2^2\;\,$$

where $\mathbf{m}_k$ refers to the model at $k$-th iteration, and $\mathbf{W}_k$ is a diagonal weight, which can be defined, for example, as $\mathbf{W}_0=\mathbf{I}$ and

$$\mathbf{W}_k = \mbox{diag}\left[\sqrt{|\mathbf{m}_{k-1}|}\right]$$

for $k > 0$. Each iteration requires a regular least-squares inversion and can be accomplished using the conjugate-gradient algorithm. Thus, the iteration consists of cycles of inner conjugate-gradient iterations and outer reweighting iterations. If the outer iteration converges, the weighted $L_2$ norm will approach the $L_1$ norm.

To avoid division by zero and accelerate the convergence, we can also rewrite the objective function using preconditioning as follows:

$$\|\mathbf{F\,W}_k\,\mathbf{x}_k-\mathbf{d}\|_2^2 + 2\,\mu\,\|\mathbf{x}_k\|_2^2\;\,$$

with $\mathbf{m}_k = \mathbf{W}_k\,\mathbf{x}_k$.
"""

# ╔═╡ 2704579a-157e-4cea-82c8-e672ba8e7549
md"""
* Daubechies, I., R. DeVore, M. Fornasier, and C. S. Güntürk, 2010, Iteratively reweighted least squares minimization for sparse recovery: Communications on Pure and Applied Mathematics: A Journal Issued by the Courant Institute of Mathematical Sciences, 63, 1–38.
* Holland, P. W., and R. E. Welsch, 1977, Robust regression using iteratively reweighted least-squares: Communications in Statistics-theory and Methods, 6, 813–827.
* Scales, J. A., and A. Gersztenkorn, 1988, Robust methods in inverse theory: Inverse Problems, 4, 1071.
"""

# ╔═╡ d04c286a-b478-4e6f-bd4c-3614dea72112
md"""
## TV regularization

In the discussion of the smoothing operation, we described it using the diffusion equation

$$\frac{\partial u}{\partial t} = a^2\,\nabla^2 u\;.$$

One step of diffusion using implicit finite-differences

$$\frac{u(\mathbf{x},t+\Delta t) - u(\mathbf{x},t)}{\Delta t} = \,a^2\,\nabla^2 u(\mathbf{x},t+\Delta t)$$

corresponds to inversion

$$4\mathbf{u}_{t+\Delta t} = \left(\mathbf{I} +
  \epsilon^2\,\mathbf{G}^T\,\mathbf{G}\right)^{-1}\,\mathbf{u}_t\;,$$

where $\mathbf{u}_t$ represents $u(\mathbf{x},t)$, $\mathbf{G}$ stands for the gradient operator, and $\epsilon = \Delta t\,a^2$. This equation can be understood as the result
of the least-squares minimization of 

$$\|\mathbf{u}_{t+\Delta t} - \mathbf{u}_t\|^2 +
\epsilon^2\,\|\mathbf{G\,u}_{t+\Delta t}\|^2\;.$$

What happens if we change the $L_2$ norm in the second term to the $L_1$ norm (known as *total variation*)? Following the logic of IRLS, this may amount to weighting the gradient by the square root of its absolute value:

$$\|\mathbf{u}_{t+\Delta t} - \mathbf{u}_t\|^2 + \epsilon^2\,\|\mathbf{W\,G\,u}_{t+\Delta t}\|^2\;.$$

Correspondingly, the linear diffusion equation will be transformed to the nonlinear equation

$$\frac{\partial u}{\partial t} = \displaystyle a^2\,\nabla \cdot \left(\frac{\nabla
  u}{\left|\nabla u\right|}\right)\;.$$

Scaling the gradient in that manner corresponds to a different form of smoothing, which distinguishes between small and large gradients in the input and preserves edges (places of large gradient). 
"""

# ╔═╡ 2030e50e-ae91-428b-bae0-c46ff685693b
download("https://ahay.org/data/hall/horizon.asc","horizon.asc")

# ╔═╡ 842d0065-11b6-4818-a6b6-c9c1a78b38b2
xyz = readdlm("horizon.asc"); # read data from a text file

# ╔═╡ a3c9563a-1b3e-4d51-add8-7c5910192fea
begin
	n1, n2 = 196, 291
	iline = xyz[1:n1:end, 1]
	xline = xyz[1:n1, 2]
end

# ╔═╡ ce344c35-a234-42e8-a4c6-e467e58770c1
begin
	horizon=reshape(xyz[:, 3], (n1, n2))
	# subtract mean
	mean = sum(horizon)/length(horizon)
	horizon .-= mean
end

# ╔═╡ 60d3f5b5-3102-4ec3-adc4-91430d14916a
plot_horizon(map, title) = heatmap(iline, xline, map, 
		title=title, xlabel=L"$x_1$ (m)", ylabel=L"$x_2$ (m)",
	    clim=(-14,14))

# ╔═╡ c9a9e870-e929-4a7b-bd82-b86e604b65d2
plot_horizon(horizon, "Seismic Horizon")

# ╔═╡ df9da995-3180-4306-8c87-f3317fb187ae
function smooth(x::Vector{T}, nb::Int) where T <: Real
    "triangle smoothing by explicit diffusion"
    n = length(x)
    z, y = similar(x), copy(x) 
    for ib in 1:nb-1
        α = 1/(4*sin(π*ib/nb)^2)
        # reflecting boundary conditions
        z[1] = (1 - 2*α)*y[1] + α*(y[1] + y[2])
        for i in 2:n-1
            z[i] = (1 - 2*α)*y[i] + α*(y[i-1] + y[i+1])
        end
        z[n] = (1 - 2*α)*y[n] + α*(y[n-1] + y[n])
        y, z = z, y
    end
    return y 
end

# ╔═╡ 69bee66b-f653-4274-ae2e-cc3fa969fc67
smoothed = mapslices(
					 slice -> smooth(slice, 20), 
                     mapslices(
						 slice -> smooth(slice, 20), 
						 horizon; 
						 dims=1); 
					 dims=2);

# ╔═╡ 8006cd1e-8fb9-4fe1-99c9-9959d7cb5c66
plot_horizon(smoothed, "Diffusion")

# ╔═╡ 67999f22-32e6-44fa-b595-ba707e12037b
md"""
Regular smoothing by triangle filtering helps remove the noise but smears the edges of the channel. We can visualize the edges using an edge detection algorithm.

[https://en.wikipedia.org/wiki/Canny\_edge\_detector](https://en.wikipedia.org/wiki/Canny_edge_detector)
"""

# ╔═╡ 18f8721e-64a5-4aa3-9282-0a7b16981be9
function sobel(x::Matrix)
	"Sobel gradient filter"
	n1, n2 = size(x)
	w1 = zeros(eltype(x), n1, n2)
	w2 = zeros(eltype(x), n1, n2)
	@inbounds for i1 in 2:n1-1, i2 in 2:n2-1
		w1[i1,i2] =
		    x[i1+1,i2+1] - x[i1-1,i2+1] +
		    2*(x[i1+1,i2] - x[i1-1,i2]) +
		    x[i1+1,i2-1] - x[i1-1,i2-1]; 
		w2[i1,i2] =
		    x[i1+1,i2+1] - x[i1+1,i2-1] +
		    2*(x[i1,i2+1] - x[i1,i2-1]) +
		    x[i1-1,i2+1] - x[i1-1,i2-1]
	end
	return w1, w2
end

# ╔═╡ cea7a865-aa15-4e1b-9777-7769b3781f5d
function canny(x::Matrix; pmax=0.95, pmin=0.05)
	"Canny edge detector"
	w1, w2 = sobel(x)
	ww = w1 .* w1 + w2 .* w2
	pp = similar(ww)
	n1, n2 = size(ww)
	edge = Array{Int8}(undef, n1, n2)
	# edge thinning 
	for i1 in 1:n1, i2 in 1:n2
		g1 = w1[i1,i2]
		g2 = w2[i1,i2]
		if (abs(g1) > abs(g2)) 
		    j1=1
		    if (g2/g1 > 0.5)  
				j2=1
			elseif (g2/g1 < - 0.5) 
				j2=-1 
		    else
				j2=0
			end
		elseif (abs(g2) > abs(g1)) 
		    j2=1
		    if (g1/g2 > 0.5)  
				j1=1
		    elseif (g1/g2 < - 0.5) 
				j1=-1 	
		    else 
				j1=0
			end
		else 
		    j1=0
		    j2=0
		end
		
		k1 = i1+j1
		if (k1 < 1 || k1 > n1)
			k1=i1
		end
		k2 = i2+j2
		if (k2 < 1 || k2 > n2) 
			k2=i2
		end
		if (ww[i1,i2] <= ww[k1,k2]) 
		    pp[i1,i2] = zero(eltype(pp))
			continue
		end
		
		k1 = i1-j1
		if (k1 < 1 || k1 > n1)
			k1=i1
		end
		k2 = i2-j2
		if (k2 < 1 || k2 > n2) 
			k2=i2
		end
		if (ww[i1,i2] <= ww[k1,k2]) 
		    pp[i1,i2] = zero(eltype(pp))
			continue
		end		

		pp[i1,i2] = ww[i1,i2]
	end
	# edge selection 
	wmax = quantile(ww[:],pmax)
	wmin = quantile(ww[:],pmin)
	nedge=0
	for i1 in 1:n1, i2 in 1:n2
		w = pp[i1,i2]
		if (w > wmax) 
		    edge[i1,i2] = 1 
		    nedge += 1
		elseif (w < wmin) 
		    edge[i1,i2] = 0 
		else
		    edge[i1,i2] = 2 
		end
	end
	nold=0
	while (nedge != nold) 
	    nold = nedge
		for i1 in 1:n1, i2 in 1:n2
		    if (2 == edge[i1,i2]) 
				if (i2 > 1) 
				    if (1 == edge[i1,i2-1] || 
					(i1 > 1  && 1 == edge[i1-1,i2-1]) ||
					(i1 < n1 && 1 == edge[i1+1,i2-1])) 
						edge[i1,i2] = 1
						nedge += 1
						continue
					end
				end
				if (i2 < n2) 
			    	if (1 == edge[i1,i2+1] || 
					(i1 > 1  && 1 == edge[i1-1,i2+1]) ||
					(i1 < n1 && 1 == edge[i1+1,i2+1])) 
						edge[i1,i2] = 1
						nedge += 1
						continue
					end
				end
				if ((i1 > 1  && 1 == edge[i1-1,i2]) ||
			    	(i1 < n1 && 1 == edge[i1+1,i2])) 
			    	edge[i1,i2] = 1
			    	nedge += 1
				end
			end
		end
	end
	for i1 in 1:n1, i2 in 1:n2
		if (2 == edge[i1,i2]) 
			edge[i1,i2] = 0
		end
	end
	return edge
end

# ╔═╡ ae92bab2-5bd9-4d03-9ed6-5df0dd79eafb
edges = canny(horizon);

# ╔═╡ 6e8d387c-7da1-4701-b23d-76182174e53f
# mask for edges
edge_mask(x) = x > 0 ? x : NaN32

# ╔═╡ 0d6102e8-1d29-45bc-bf19-6b1e6bb649d7
begin
	plot_horizon(horizon, "Seismic Horizon with Edges")
	heatmap!(iline, xline, edge_mask.(edges), cmap=:grays, cbar=:none)
end

# ╔═╡ fbaead87-25b1-45a7-86fa-2b0f7109ea20
begin
	plot_horizon(smoothed, "Smoothed Seismic Horizon with Edges")
	smooth_edges = canny(smoothed)
	heatmap!(iline, xline, edge_mask.(smooth_edges), cmap=:grays, cbar=:none)
end

# ╔═╡ 343d82f7-5114-44e8-a258-b870fca45574
md"""
* Rudin, L. I., S. Osher, and E. Fatemi, 1992, Nonlinear total variation based noise removal algorithms: Physica D: Nonlinear Phenomena, 60, 259–268.
* Weickert, J., 1998, Anisotropic diffusion in image processing: Teubner.
"""

# ╔═╡ c649156a-b81d-4ab4-94a4-f8f94d124426
md"""
## Bilateral filtering

Another effective approach to edge-preserving smoothing is
*bilateral filtering*.

Regular smoothing can be
described theoretically with the help of a Gaussian weight

$$u_s(\mathbf{x}) = A\,\int u(\mathbf{y})\, \exp{\left(-\frac{|\mathbf{x-y}|^2}{r^2}\right)}\,d\mathbf{y}\;,$$

where $u(\mathbf{y})$ is the input function, $u_s(\mathbf{x})$ is its smoothed version, $r$ is the smoothing radius, and $A$ is the normalization

$$A = \left(\int \exp{\left(-\frac{|\mathbf{x}|^2}{r^2}\right)}\,d\mathbf{x}\right)^{-1}\;.$$
"""

# ╔═╡ c0b4e88f-7712-4155-8653-d66f2e13c447
md"""
In comparison, bilateral smoothing is a nonlinear operation

$$u_b(\mathbf{x}) = B(\mathbf{x})\,\int u(\mathbf{y})\,
  \exp{\left(-\frac{|\mathbf{x-y}|^2}{r^2}
    -\frac{|u(\mathbf{x})-u(\mathbf{y}|^2}{\rho^2}\right)}\,d\mathbf{y}\;,$$

where

$$B(\mathbf{x}) = \left(\int 
  \exp{\left(-\frac{|\mathbf{x-y}|^2}{r^2}
      -\frac{|u(\mathbf{x})-u(\mathbf{y}|^2}{\rho^2}\right)}\,d\mathbf{y}\right)^{-1}\;.$$

In other words, we determine the neighborhood used for local
averaging not only by the distance from the center but also by the difference in data values. Across an edge, the latter is large, and smoothing decreases, as controlled by the parameter $\rho$.
"""

# ╔═╡ 2826957f-544a-41ec-b846-c646df501a7c
md"""
* Tomasi, C., and R. Manduchi, 1998, Bilateral filtering for gray and color images: Sixth international conference on computer vision (IEEE Cat. No. 98CH36271), IEEE, 839–846.
"""

# ╔═╡ 88c01211-c09e-4e78-aacc-35c739185d6b
function bilateral(image::Array, r::Vector{Int}, a::Vector) 
	n1, n2 = size(image)
	smoothed = similar(image)
	nul = zero(eltype(image))
	@inbounds for i1 in 1:n1, i2 in 1:n2
		smoo = norm = nul
		im0 = image[i1, i2]
		for k1 in -r[1]:r[1], k2 in -r[2]:r[2]
			j1, j2 = i1 + k1, i2 + k2
			if 1 <= j1 <= n1 && 1 <= j2 <= n2
				im = image[j1, j2]
				diff = im - im0
				gauss = exp(-a[1]*k1*k1 - a[2]*k2*k2 - a[3]*diff*diff)
				smoo += im * gauss
				norm += gauss
			end
		end
		smoothed[i1, i2] = norm > nul ? smoo/norm : nul
	end
	return smoothed
end

# ╔═╡ 325ca66e-a4ce-485f-ac18-eadd5e23abfc
bsmoothed = bilateral(horizon, [10, 10], [0.01, 0.01, 0.001]);

# ╔═╡ f04dc62b-9868-4939-a05b-ce39cabf955a
begin
	plot_horizon(bsmoothed, "Bilateral Filtering with Edges")
	bsmooth_edges = canny(bsmoothed)
	heatmap!(iline, xline, edge_mask.(bsmooth_edges), cmap=:grays, cbar=:none)
end

# ╔═╡ 05559c11-81b8-4b2e-939f-2e2b14d63cc6
md"""
!!! assignment 

    ## Task 3
    Add another parameter to the `bilateral` function to apply bilateral smoothing multiple times. Test whether repeated smoothing improves the result.
"""

# ╔═╡ 062efaf9-a89d-478f-a96f-9879e2a3c1f8
md"""
!!! assignment

    ## Task 4

    Using the `BenchmarkTools` package, measure the computational efficiency of bilateral filtering and compare it with that of alternative methods.
"""

# ╔═╡ 7d0ca600-309a-400c-98a7-ccf5be1e5e1b
md"""
!!! assignment

    ## Bonus Task

    Improve the performance of the `bilateral` function by modifying the code.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUIExtra = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
FFTW = "~1.10.0"
HTTP = "~1.11.0"
LaTeXStrings = "~1.4.0"
Plots = "~1.41.3"
PlutoUIExtra = "~0.1.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "fa9e9391c95d2cc66a9bfeb9786c86ead9eea0e9"

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
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

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
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

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
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

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
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

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
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccc81ba5e42497f4e76553a5545665eed577a663"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.0.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

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
git-tree-sha1 = "88fb6ab75454c21be1d75a0a430a0ed95f0d3f1e"
uuid = "6394faf6-06db-4fa8-b750-35ccc60383f7"
version = "0.1.28"

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
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "b7bfd56fa66616138dfe5237da4dc13bbd83c67f"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "f305bdb91e1f3fcc687944c97f2ede40585b1bd5"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.19"

    [deps.GR.extensions]
    GRIJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "de439fbc02b9dc0e639e67d7c5bd5811ff3b6f06"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.19+1"

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
git-tree-sha1 = "6b4d2dc81736fe3980ff0e8879a9fc7c33c44ddf"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.2+0"

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
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "5b6bb73f555bc753a6153deec3717b8904f5551c"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.3.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6893345fd6658c8e475d40155789f4860ac3b21"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.4+0"

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
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

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
git-tree-sha1 = "3acf07f130a76f87c041cfb2ff7d7284ca67b072"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.2+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2a7a12fc0a4e7fb773450d17975322aa77142106"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.2+0"

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
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

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

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

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
git-tree-sha1 = "0662b083e11420952f2e62e17eddae7fc07d5997"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

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
git-tree-sha1 = "459d8913a8b83c7222eb629664283653dadfe2b6"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.3"

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
git-tree-sha1 = "0d751d4ceb9dbd402646886332c2f99169dc1cfd"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.76"

[[deps.PlutoUIExtra]]
deps = ["AbstractPlutoDingetjes", "ConstructionBase", "FlexiMaps", "HypertextLiteral", "InteractiveUtils", "IntervalSets", "Markdown", "PlutoUI", "Random", "Reexport"]
git-tree-sha1 = "b4ff5d24e2dc8fbf319cd44f9f81b5356e27bafb"
uuid = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"
version = "0.1.8"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

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
git-tree-sha1 = "34f7e5d2861083ec7596af8b8c092531facf2192"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "8f528b0851b5b7025032818eb5abbeb8a736f853"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+2"

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

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

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
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "be5733d4a2b03341bdcab91cea6caa7e31ced14b"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.9"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "79529b493a44927dd5b13dde1c7ce957c2d049e4"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.0"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
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
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

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
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

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
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

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
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

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
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

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
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

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
git-tree-sha1 = "de8ab4f01cb2d8b41702bab9eaad9e8b7d352f73"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.53+0"

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
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

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
# ╟─668405bb-7ade-4455-8697-34b1747f0b26
# ╟─cb9e8b2b-8346-4696-a3c3-d335ec246afc
# ╟─6632dc0e-b2c9-4294-96a3-e008d3b98d01
# ╟─14a0572a-f427-4103-ae9d-06fd0ce2ef0f
# ╟─b0da32b2-aa83-4f9c-9456-1caa60baeab3
# ╟─d7bb2b15-413e-4631-9de5-a961d7103611
# ╠═9bd5832e-67a2-4ccf-9a39-712ebfc46cdb
# ╠═748715c1-5a28-4b07-ba48-70b99880f9cd
# ╠═92108c6b-948f-4cba-9ba1-d00d89a24094
# ╟─3b15ba8c-1d8e-4660-93ec-6ed1afc1fdce
# ╠═c80d2f03-2bb1-4ba3-86e6-b3365a9e0a73
# ╠═24822111-42f6-4f14-9cb4-1344bee1de86
# ╠═d4545fc9-16f1-41c6-8cf9-1597eb3f38ea
# ╠═743bf67f-a2fd-46ee-8a43-e28dd3f43c7e
# ╠═09851753-bfa7-4c7c-9502-0937835d9240
# ╠═bb48142b-4daf-4afe-8e74-5bd184f794a1
# ╠═fbb2d10f-2001-446c-a108-456330a98812
# ╠═075920c6-5496-4ebe-a77e-fca3fe205d78
# ╠═3f293e34-6b6e-4841-8543-84ae12fe4ed5
# ╠═e58e1fa1-d0f5-4151-a037-5b42dfc10e55
# ╟─09541de5-56ba-4050-90b7-75a11edba450
# ╠═a4ef85a2-7a41-44a2-9541-ebb65758f99e
# ╠═9594ef17-371c-4803-9f39-5ae94e9e879e
# ╠═c51884ae-90af-4fe0-83c9-aee5a8399d5b
# ╠═451655cb-2926-44ca-aab1-e12a2bf21f17
# ╟─7b205fa2-af01-477f-9aec-a392965b998c
# ╟─f4927c25-652c-49bb-af20-bb690ddf3dd8
# ╟─d16aebc9-209f-43de-8aaf-e0f5dd7dbe6d
# ╟─15b094f1-2f1c-48b4-a78e-65428c3e882a
# ╟─00c547fd-36dc-403d-be66-905259734ac1
# ╟─a1056c94-49d2-40db-9926-c15fa90b6357
# ╟─a5be1a5b-0382-4e21-9ed5-d5d33a95b5bb
# ╟─2a3c0498-f714-4dad-b6d7-d1a049d580d3
# ╟─38af1498-ce38-4ea9-baec-51dfc3c78e45
# ╠═e8365393-de2c-4a47-8ee3-b7d22e341033
# ╠═e289b562-c16b-45e4-9ed0-47c90adf3371
# ╠═062fd5cb-42ae-4845-93cd-d988a071d154
# ╠═8f92b618-b37f-4061-82a9-37639598dbeb
# ╠═a8c11790-7e42-4880-99c8-ce349d6deec6
# ╠═cc83782b-2d84-4db6-b9d9-1a08cbe4a066
# ╠═1cb1a5a4-d3aa-41b2-9ced-82063d3790cf
# ╠═a2fca1ec-2a6a-49ad-9284-aea3811d0478
# ╠═51cdb012-ff83-40fe-a142-aeed7d57f703
# ╠═74ae169c-d1be-4972-a0be-d3ac58d81541
# ╠═fb2a7ac8-a6b6-4015-8ec3-01f43ff0f36c
# ╠═302e9c8b-f286-494a-aaec-9f1be0afc435
# ╟─b7095ce7-1d91-477b-9b8f-82f56da5176d
# ╟─7797bff3-94a4-4635-b7f6-ad959f26560c
# ╟─2704579a-157e-4cea-82c8-e672ba8e7549
# ╟─d04c286a-b478-4e6f-bd4c-3614dea72112
# ╠═ef693662-cfa4-4dd6-9723-8d6485d03f6e
# ╠═2030e50e-ae91-428b-bae0-c46ff685693b
# ╠═842d0065-11b6-4818-a6b6-c9c1a78b38b2
# ╠═a3c9563a-1b3e-4d51-add8-7c5910192fea
# ╠═ce344c35-a234-42e8-a4c6-e467e58770c1
# ╠═25684269-c636-46a0-88fa-1bf71fa457a4
# ╠═60d3f5b5-3102-4ec3-adc4-91430d14916a
# ╠═c9a9e870-e929-4a7b-bd82-b86e604b65d2
# ╠═df9da995-3180-4306-8c87-f3317fb187ae
# ╠═69bee66b-f653-4274-ae2e-cc3fa969fc67
# ╠═8006cd1e-8fb9-4fe1-99c9-9959d7cb5c66
# ╟─67999f22-32e6-44fa-b595-ba707e12037b
# ╠═18f8721e-64a5-4aa3-9282-0a7b16981be9
# ╠═b5118e1d-5263-47c8-8e91-e90f9edaafae
# ╠═cea7a865-aa15-4e1b-9777-7769b3781f5d
# ╠═ae92bab2-5bd9-4d03-9ed6-5df0dd79eafb
# ╠═6e8d387c-7da1-4701-b23d-76182174e53f
# ╠═0d6102e8-1d29-45bc-bf19-6b1e6bb649d7
# ╠═fbaead87-25b1-45a7-86fa-2b0f7109ea20
# ╟─343d82f7-5114-44e8-a258-b870fca45574
# ╟─c649156a-b81d-4ab4-94a4-f8f94d124426
# ╟─c0b4e88f-7712-4155-8653-d66f2e13c447
# ╟─2826957f-544a-41ec-b846-c646df501a7c
# ╠═88c01211-c09e-4e78-aacc-35c739185d6b
# ╠═325ca66e-a4ce-485f-ac18-eadd5e23abfc
# ╠═f04dc62b-9868-4939-a05b-ce39cabf955a
# ╟─05559c11-81b8-4b2e-939f-2e2b14d63cc6
# ╟─062efaf9-a89d-478f-a96f-9879e2a3c1f8
# ╟─7d0ca600-309a-400c-98a7-ccf5be1e5e1b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
