### A Pluto.jl notebook ###
# v0.20.27

#> [frontmatter]
#> chapter = 4
#> section = 16
#> order = 16
#> image = "https://raw.githubusercontent.com/fonsp/Pluto.jl/580ab811f13d565cc81ebfa70ed36c84b125f55d/demo/plutodemo.gif"
#> title = "16 Data Analysis Notebook"
#> tags = ["module4", "track_julia", "track_material", "Pluto", "PlutoUI"]
#> layout = "layout.jlhtml"

using Markdown
using InteractiveUtils

# ╔═╡ 7e40faf0-a503-4c98-a9d2-9a824a594768
begin
	using PlutoUIExtra
	TableOfContents()
end

# ╔═╡ adaaeb9e-2cef-4667-bf78-3854bf9271e2
using LinearAlgebra # defines dot-product as "⋅"

# ╔═╡ 0f28e9f6-5670-46bc-a1a5-11cb6dbbad39
using Plots

# ╔═╡ a662f040-1375-46c4-b9fc-0998dc6d357b
using Random

# ╔═╡ 1d477149-313a-4265-b498-09ba3c0eafd4
using BenchmarkTools 

# ╔═╡ 8164a2e1-467a-4a4b-9240-f28da7df2b40
using DelaunayTriangulation

# ╔═╡ 5f3e2b21-2a7b-49b1-af1e-01272da47c58
using IterativeSolvers, LinearMaps # for gmres

# ╔═╡ cdc06c3e-c3ae-491a-b974-f936c95af886
import HTTP

# ╔═╡ 613746d8-8817-4160-bb29-b3ceaa0cd0c2
let
    notebook_dir = @__DIR__
    toc_path = joinpath(notebook_dir, "TOC_Notebook.jl")
    
    Markdown.parse("[⬅️ Back to Table of Contents](./open?path=$(HTTP.escapeuri(toc_path)))")
end

# ╔═╡ 5466a099-b57c-4a7c-aea0-91a258d834d4
md"""
# Shaping regularization

So far, our discussion of estimation problems has focused on the optimization (least-squares) framework within statistical linear estimation theory. This chapter will examine an alternative framework known as *shaping regularization*. We will derive a different form of the estimation operator and then uncover connections between the shaping and optimization frameworks.
"""

# ╔═╡ 20d7074b-7daa-420f-81a7-7bc63fa10a1c
md"""
## Going forward and backward

As before, suppose that the data (what we measure) $\mathbf{d}$ is
connected to the model (what we want) $\mathbf{m}$ through the forward
operator $\mathbf{F}$:

$$\mathbf{d = F\,m}\;.$$

Also, suppose that we can construct an efficient backward
(approximate inverse) operator for going back from the data space to
the model space

$$\tilde{\mathbf{m}} = \mathbf{B\,d}\;.$$

A particular example of $\mathbf{B}$ is the adjoint operator
$\mathbf{F}^T$ in the linear case or the adjoint of the derivative of
$\mathbf{F}$ if $\mathbf{F}$ is nonlinear. However, the concept is
more general.

The pair of $\mathbf{F}$ and $\mathbf{B}$ allows for the following
constructive iterative inversion procedure. Putting the forward and backward equations together, we arrive at the equation

$$\tilde{\mathbf{m}} = \mathbf{B\,F\,m}\;,$$

which can be solved by simple iteration

$$\mathbf{m}_{n+1} = \mathbf{m}_{n} + \tilde{\mathbf{m}} -  \mathbf{B\,F\,m}_n = \tilde{\mathbf{m}} + (\mathbf{I - B\,F})\,\mathbf{m}_n\;,$$

starting with some $\mathbf{m}_{0}$.  The iteration
converges to a solution of the original equation provided that
$\mathbf{B\,F}$ is sufficiently close to the identity operator.

Alternatively, we could make a chain $\mathbf{F\,B}$ and iterate

$$\mathbf{d}_{n+1} = \mathbf{d}_{n} + \mathbf{d} -  \mathbf{F\,B\,d}_n = \mathbf{d} + (\mathbf{I - F\,B})\,\mathbf{d}_n\;,$$

starting with $\mathbf{d}_0 = \mathbf{d}$ and completing the iteration
by computing $\mathbf{m}_N = \mathbf{B\,d}_N$.
"""

# ╔═╡ c3570349-fdee-490f-b1d6-0f13daf82a8d
md"""
## Introducing shaping

Not all information about the model is contained in the data. Additional constraints may come from other sources of information. Let us introduce a *model shaping* operator $\mathbf{S}_m$. Shaping is applied to an estimated model to transform it into a model that satisfies additional constraints. Model shaping can be incorporated into the iteration as follows:

$$\mathbf{m}_{n+1} = \mathbf{S}_m\left[\tilde{\mathbf{m}} + (\mathbf{I - B\,F})\,\mathbf{m}_n\right]\;.$$

If $\mathbf{F}$, $\mathbf{B}$, and $\mathbf{S}_m$ are linear and if
 the iteration converges to $\mathbf{m}_{\infty}$, then
the following condition is satisfied:

$$\mathbf{m}_{\infty} = \mathbf{S}_m\left[\tilde{\mathbf{m}} + (\mathbf{I - B\,F})\,\mathbf{m}_{\infty}\right]$$

or

$$\mathbf{m}_{\infty} = \left[\mathbf{I} + \mathbf{S}_m\,(\mathbf{B\,F - I})\right]^{-1}\,\mathbf{S}_m\,\tilde{\mathbf{m}}
= \left[\mathbf{I} + \mathbf{S}_m\,(\mathbf{B\,F - I})\right]^{-1}\,\mathbf{S}_m\,\mathbf{B\,d}\;.$$

Alternatively, we can incorporate data-domain constraints into the data-space iteration via a *data shaping* operator $\mathbf{S}_d$. The iteration becomes

$$\mathbf{d}_{n+1} = \mathbf{S}_d\left[\mathbf{d}_{n} + \mathbf{d} -  \mathbf{F\,B\,d}_n\right]\;.$$

If it converges and if $\mathbf{F}$, $\mathbf{B}$, and $\mathbf{S}_d$
are linear operators, then, analogously to the model-space iteration,

$$\mathbf{d}_{\infty} = \left[\mathbf{I} + \mathbf{S}_d\,(\mathbf{F\,B - I})\right]^{-1}\, \mathbf{S}_d\,\mathbf{d}$$

and

$$\mathbf{m}_{\infty} = \mathbf{B}\,\left[\mathbf{I} + \mathbf{S}_d\,(\mathbf{F\,B - I})\right]^{-1}\, \mathbf{S}_d\,\mathbf{d}\;.$$

The model-space and data-space inversion equations complete the definition of shaping regularization as an alternative to optimization.
"""

# ╔═╡ e8a121b1-fff3-405c-989d-f6969cc5da6e
md"""
## Connection with least-squares optimization

At first glance, the shaping equations look very different from the estimation equations we used earlier. However, their algebraic equivalence can be established under certain conditions. In particular, the model-space equation becomes algebraically equivalent to our previous equation

$\widehat{\mathbf{m}} = \left(\mathbf{F}^T\,\mathbf{C}_n^{-1}\,\mathbf{F} + \mathbf{C}_{m}^{-1}\right)^{-1}\,\mathbf{F}^T\,\mathbf{C}_n^{-1}\,\mathbf{d}$

if we make the following choices:

$\begin{array}{rcl}
  \mathbf{B} & = & \mathbf{F}^T\,\mathbf{C}_n^{-1}\;, \\
  \mathbf{S}_m & = & \left(\mathbf{I} + \mathbf{C}_m^{-1}\right)^{-1}\;.\end{array}$
"""

# ╔═╡ 028eda0f-4abb-42f0-891a-a0e646982814
md"""
Analogously, the data-space shaping is equivalent to the previous equation

$\widehat{\mathbf{m}} = \mathbf{C}_{m}\,\mathbf{F}^T\,\left(\mathbf{F}\,\mathbf{C}_{m}\,\mathbf{F}^T + \mathbf{C}_n\right)^{-1}\,\mathbf{d}$

if we choose

$\begin{array}{rcl}
  \mathbf{B} & = & \mathbf{C}_m\,\mathbf{F}^T\;, \\
  \mathbf{S}_d & = & \left(\mathbf{I} + \mathbf{C}_n\right)^{-1}\;.\end{array}$

The algebraic equivalence is proven by direct substitution. 

Why would we prefer a particular expression if an alternative produces algebraically equivalent results? To make a practical choice, we need to decide: 

1. Which operator is easier to specify ($\mathbf{C}_m$, $\mathbf{C}_m^{-1}$, or $\mathbf{S}_m$; $\mathbf{C}_d$, $\mathbf{C}_d^{-1}$, or $\mathbf{S}_d$)?
2. Which matrix is easier to invert (taking fewer iterations with an iterative method such as conjugate gradients)? 

Different data analysis applications may lead to different choices.
"""

# ╔═╡ 504b099f-8bb9-455e-8e5a-2300bd6de853
md"""
## Proximity operators

Is there a deeper meaning to choosing model shaping in the form $\left(\mathbf{I} + \mathbf{C}_m^{-1}\right)^{-1}$? Notice that this operator resembles the estimation operator for the case $\mathbf{F}=\mathbf{C}_n=\mathbf{I}$. Using the previously established connection to least-squares inversion, we can say that model shaping in the form

$$\mathbf{m} = \left(\mathbf{I} + \mathbf{C}_m^{-1}\right)^{-1}\,\mathbf{m}_0$$

results from minimizing the least-squares objective function

$$\left|\mathbf{m}-\mathbf{m}_0\right|^2+\left|\mathbf{D}_m\,\mathbf{m}\right|^2\,$$

with $\mathbf{D}_m^T\,\mathbf{D}_m = \mathbf{C}_m^{-1}$.

Analogously, the data-shaping operator 

$$\mathbf{d} = \left(\mathbf{I} + \mathbf{C}_n\right)^{-1}\,\mathbf{d}_0$$

can be understood as the result of minimizing $|\mathbf{d}|^2+|\mathbf{x}|^2$ for $\mathbf{d}$ and $\mathbf{x}$ under the constraints

$$\mathbf{d} + \mathbf{P}_n\,\mathbf{x} = \mathbf{d}_0$$

and $\mathbf{P}_n\,\mathbf{P}_n^T = \mathbf{C}_n$.

For a general convex functional $R$ applied to the model, the *proximity operator* $\mathbf{P}_R$ is defined as the following minimizer:

$$\mathbf{P}_R(\mathbf{m}_0) = \arg \min_{\mathbf{m}} \left[\left|\mathbf{m}-\mathbf{m}_0\right|^2+R(\mathbf{m})\right]\;.$$

The proximity operator shapes the input model $\mathbf{m}_0$ by penalizing unwanted features. Thus, the shaping operator is the proximity operator for $R(\mathbf{m}) = \left|\mathbf{D}_m\,\mathbf{m}\right|^2$. The general notion of proximity operators extends the connection between the optimization framework and the shaping framework to the case of nonlinear regularizations.
"""

# ╔═╡ 43ad1b2b-1fc1-48af-881f-c1daf33e47be
md"""
## Shaping conjugate gradients

The versions of conjugate gradients that we considered before was concerned with solving the problem of minimizing $\left|\mathbf{A}\,\mathbf{x}-\mathbf{b}\right|^2$, equivalent to solving $\mathbf{A}^T\,\mathbf{A\,x}=\mathbf{A}^T\,\mathbf{b}$. If we
think of conjugate gradients as a method of solving the general linear system $\mathbf{B\,x}=\mathbf{d}$ with a positive definite matrix $\mathbf{B}$, the algorithm takes the form implemented below.
"""

# ╔═╡ 7168dd81-1f22-412a-954e-3f9c5c3f050c
function conjgrad(operator::Function, d::Array, x0::Array, niter::Int)
    "Conjugate-gradient algorithm for solving B x = d"
    x = deepcopy(x0)
    g = operator(x) - d
    s, S = similar(x), similar(d)
    gnp = zero(eltype(x))
    for iter in 1:niter
        G = operator(g)
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
        α = -gn/(S ⋅ s)
        x += α*s
        g += α*S
    end
    return x
end

# ╔═╡ 95b3b4f3-c9dd-41ff-a8f6-dd33b39d2018
md"""
The matrix inverted in the shaping equation is not symmetric. However, if we assume the possibility of the symmetric splitting $\mathbf{S}_m=\mathbf{H}\,\mathbf{H}^T$ and the invertability of $\mathbf{H}$, we can rewrite the equation in a more symmetric form, as follows:

$$\mathbf{m}_{\infty} 
= \left[\mathbf{I} + \mathbf{S}_m\,(\mathbf{B\,F - I})\right]^{-1}\,\mathbf{S}_m\,\mathbf{B\,d} 
= \mathbf{H}\,\left[\mathbf{I} + \mathbf{H}^T\,(\mathbf{B\,F - I})\,\mathbf{H}\right]^{-1}\,\mathbf{H}^T\,\mathbf{B\,d}\;.$$

If, additionally, we select the backward operator $\mathbf{B}$ as $\mathbf{B}=\mathbf{F}^T\,\mathbf{C}_n^{-1}$, the inverted matrix becomes suitable for an application of conjugate gradients.
"""

# ╔═╡ 18ac4fcb-6e6e-40bc-8e58-07a766ede06d
md"""
!!! assignment
    ## Task 1 (theoretical)

    Assuming a symmetric splitting for the data shaping operator $\mathbf{S}_d=\mathbf{H}_d^T\,\mathbf{H}_d$, find a symmetric form of the data shaping equation

    $\mathbf{m_\infty} = \mathbf{B}\,\left[\mathbf{I} + \mathbf{S}_d\,(\mathbf{F\,B - I})\right]^{-1}\, \mathbf{S}_d\,\mathbf{d} =$
"""

# ╔═╡ 41ebe5b1-15da-4f88-ae91-f9c412d15ca5
function conjgrad(forward::Function, adjoint::Function, shaping::Function, 
	              d::Array, p0::Array, niter=1, ϵ=1.0)
	"Conjugate-gradient algorithm for shaping regularization"
	p = deepcopy(p0)
	x = shaping(p)
	r = forward(x) .- d      
	sp, sx, sr = similar(p), similar(x), similar(r)
	gnp, g0 = zero(Float64), zero(Float64)
	for iter in 1:niter
		gx = adjoint(r) - ϵ*x
		gp = shaping(gx) + ϵ*p
		gx = shaping(gp)
		gr = forward(gx)

		gn = gp ⋅ gp
		@show iter, gn

		if iter==1
			g0 = gn

			sp = gp
			sx = gx
			sr = gr
		else
			γ = gn/g0
			β = gn/gnp
			
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

# ╔═╡ b89a90b1-643c-4b4d-9c26-cd1779ea8c78
md"""
The code implements the conjugate-gradient algorithm for finding the solution in case of $\mathbf{B}=\displaystyle \mathbf{F}^T/\epsilon$, which corresponds to

$$\mathbf{m}_{\infty} = \mathbf{H}\,\left[\epsilon\,\mathbf{I} + \mathbf{H}^T\,(\mathbf{F}^T\,\mathbf{F} - \epsilon\,\mathbf{I})\,\mathbf{H}\right]^{-1}\,\mathbf{H}^T\,\mathbf{B\,d}\;.$$

The program keeps track of three linearly dependent vectors: 
1. the auxiliary vector $\mathbf{p}$, which converges to $\mathbf{A}^{-1}\,\mathbf{H}^T\,\mathbf{F}^T\,\mathbf{d}$, where 

$$\mathbf{A} = \epsilon\,\mathbf{I} + \mathbf{H}^T\,(\mathbf{F}^T\,\mathbf{F} - \epsilon\,\mathbf{I})\,\mathbf{H}\;.$$

2. the solution vector $\mathbf{x=H\,p}$,
3. the residual vector $\mathbf{r=F\,x-d}$.

The three vectors are updated by adding steps of sizes $\mathbf{s}_p$,
$\mathbf{s}_x=\mathbf{H\,s}_p$, and $\mathbf{s}_r=\mathbf{F\,s}_p$, respectively.

To simplify computations, the program also takes note of the fact that the gradient vector is given by 

$$\mathbf{g}_p = \mathbf{A}\,\mathbf{p} - \mathbf{H}^T\,\mathbf{F}^T\,\mathbf{d}
             = \epsilon\,\mathbf{p} + \mathbf{H}^T\,(\mathbf{F}^T\,\mathbf{r} - \epsilon\,\mathbf{x})$$

and that the product $\mathbf{s}_p^T\mathbf{A}\,\mathbf{s}_p$ simplifies to

$$\mathbf{s}_p^T\mathbf{A}\,\mathbf{s}_p = \mathbf{s}_p^T\,\left[\epsilon\,\mathbf{I} + \mathbf{H}^T\,(\mathbf{F}^T\,\mathbf{F} - \epsilon\,\mathbf{I})\,\mathbf{H}\right]\,\mathbf{s}_p =
\epsilon\,\mathbf{s}_p^T\mathbf{s}_p + \mathbf{s}_r^T\mathbf{s}_r - \epsilon\,\mathbf{s}_x^T\mathbf{s}_x\;.$$

Note that each iteration of conjugate gradients in this case requires one application of $\mathbf{F}$ and $\mathbf{F}^T$ and one application of $\mathbf{H}$ and $\mathbf{H}^T$.
"""

# ╔═╡ a050e60a-d63e-4ced-9a58-70e7749e2e86
md"""
## Computational example: 1997 Spatial Interpolation Contest
"""

# ╔═╡ fef2bc4a-c36d-47be-a00b-2e0c42776e3b
import GMT

# ╔═╡ 9dec3276-49d9-4d15-82e8-ff79eb38e53a
GMT.coast(proj=:Mercator, DCW=(country="CH"), 
          title="Switzerland", show=true)

# ╔═╡ 0e518bda-c96b-41ca-a98a-5acb1f0d98e0
md"""
In 1997, the European Communities organized a Spatial Interpolation Comparison. Many organizations participated with the results published in a special issue of the *Journal of Geographic Information and Decision Analysis* and a separate report.
"""

# ╔═╡ 5c0c26ec-995c-4c8a-80a9-5668af411544
md"""
* Dubois, G., 1999, Spatial interpolation comparison 97: Foreword and introduction: Journal of Geographic Information and Decision Analysis, 2, 1–10.
* Dubois, G., J. Malczewski, and M. D. Cort, eds., 2003, Mapping radioactivity in the environment. Spatial Interpolation Comparison 1997.: Office for Official Publications of the European Communities.
"""

# ╔═╡ 28558f29-1a20-4408-8c85-cf82dd7db05d
begin
	# download data files
	download("https://ahay.org/data/rain/border.rsf@","border.bin")
	download("https://ahay.org/data/rain/alldata.rsf@","alldata.bin")
	download("https://ahay.org/data/rain/obsdata.rsf@","obsdata.bin")
end

# ╔═╡ 5e349cf8-deee-4490-a9c9-340c7ef75854
begin
	# read data
	border = Array{Float32}(undef, 2, 1289); # single-precision array
	read!("border.bin", border)
end

# ╔═╡ aae2ac2d-80da-4cd3-ae0c-d49eff6a810f
begin
	alldata = Array{Float32}(undef, 3, 467); # single-precision array
	obsdata = Array{Float32}(undef, 3, 100); # single-precision array
	read!("alldata.bin", alldata)
	read!("obsdata.bin", obsdata)
end

# ╔═╡ 81e89634-87c1-4251-a1fd-89d5a02f8f79
begin
	plot(border[1,:], border[2,:], linewidth=2, label=:none)
	scatter!(alldata[1,:], alldata[2,:], ms=2, ma=0.5, label="all stations")
	scatter!(obsdata[1,:], obsdata[2,:], markershape=:utriangle, ms=4,
	    label="test stations", title="Switzerland Weather Stations")
end

# ╔═╡ a2f3c5ad-9128-44b7-8f2d-919c3742960d
md"""
The comparison used a dataset from rainfall measurements in Switzerland on the 8th of May 1986, the day of the Chornobyl disaster. A total of 467
rainfall measurements were taken that day. A randomly selected subset of 100 measurements was used as the input data in the 1997 Spatial Interpolation Comparison to interpolate other measurements using different techniques and to compare the results with the known data. 
"""

# ╔═╡ c4fa6719-72cf-49c1-bd65-fc44e15af8b0
md"""
### Gradient regularization

Our first approach is to interpolate by solving the regularized least-squares optimization problem

$\min\left( |\mathbf{F}\,\mathbf{m} - \mathbf{d}|^2 + \epsilon^2 |\mathbf{R}\,\mathbf{m}|^2\right)\;,$

where $\mathbf{d}$ is irregular data, $\mathbf{m}$ is model estimated
on a regular grid, $\mathbf{F}$ is forward interpolation from the
regular grid to irregular locations, $\epsilon$ is a scaling
parameter, and $\mathbf{R}$ is the regularization operator related to
the inverse of the assumed model covariance. In our experiment,
$\mathbf{R}$ will be the finite-difference gradient filter.
"""

# ╔═╡ 643afb4d-2734-4037-91a9-14a6be815ba8
function conjgrad(forward::Function, adjoint::Function, regularization::Function, 
                  d::Array, x0::Array, ϵ::Real, niter::Int)
    "Conjugate-gradients for minimizing |forward(x)-d|^2 + ϵ^2*|regul(x)|^2"
    x = deepcopy(x0)
    R1 = forward(x) - d
    R2 = ϵ*regularization(x, false)
    s, S1, S2 = similar(x), similar(R1), similar(R2)
    gnp = zero(eltype(x))
    for iter in 1:niter
        g = adjoint(R1) + ϵ*regularization(R2, true) # block adjoint
        G1 = forward(g)
        G2 = ϵ*regularization(g, false)
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
		Rn = (R1 ⋅ R1 + R2 ⋅ R2) # length of the residual
		@show iter, Rn
    end
    return x
end

# ╔═╡ 7ac29fda-34b0-4f6c-8725-9a4bd1a2a09b
function dottest(forward::Function, adjoint::Function, 
                 m::Array, d::Array)
    "Dot-product test"
    mod = similar(m); rand!(mod)
    dat = similar(d); rand!(dat)
    println(" L[m]⋅d = $(forward(mod) ⋅ dat)")
    println("L'[d]⋅m = $(adjoint(dat) ⋅ mod)")
end

# ╔═╡ dd0f8427-f880-4264-8ef3-76b7e9ab593e
function gradient(x::Array)
	# finite-difference gradient operator
    n1, n2 = size(x)
	g = zeros(eltype(x), n1, n2, 2)
	@inbounds for i1 in 1:n1-1, i2 in 1:n2-1
		g[i1, i2, 1] = x[i1+1, i2] - x[i1, i2]
		g[i1, i2, 2] = x[i1, i2+1] - x[i1, i2]
	end
	return g
end

# ╔═╡ 59795e0f-fc90-4575-8ce1-8affe855096f
function gradient_adjoint(g::Array)
	# finite-difference gradient operator
    n1, n2 = size(g, 1), size(g, 2)
	x = zeros(eltype(g), n1, n2)
	@inbounds for i1 in 1:n1-1, i2 in 1:n2-1
		x[i1+1, i2] += g[i1, i2, 1]
		x[i1, i2+1] += g[i1, i2, 2]
		x[i1, i2] -= g[i1, i2, 1] + g[i1, i2, 2] 
	end
	return x
end

# ╔═╡ e57684e9-d6a1-42f5-a017-4954dcb4ddfc
rain0 = zeros(Float32, 371, 255);

# ╔═╡ 6a623beb-3dab-4172-9064-81025d599352
grad0 = zeros(Float32, 371, 255, 2);

# ╔═╡ 7d0e3bbb-8bbc-4502-825c-8c74b09c4f7e
# one-line function with one-line conditional
gradient(x::Array, adj::Bool) = adj ? gradient_adjoint(x) : gradient(x)

# ╔═╡ ac516e49-c7d0-4bf7-a0df-67cf25aec5ec
dottest(gradient, gradient_adjoint, rain0, grad0)

# ╔═╡ 991d72c3-2843-4409-b259-c3044f7c0934
function lint(regul::Array, coord; d=[1,1], o=[0,0])
    "bilinear interpolation"
    n1, n2 = size(regul)
    nd = size(coord, 2)
    irreg = zeros(eltype(regul), nd)
    for id in 1:nd
		# find nearest neighbor
		x1 = 1 + (coord[1,id] - o[1])/d[1]
		x2 = 1 + (coord[2,id] - o[2])/d[2]
        i1, i2 = floor(Int, x1), floor(Int, x2)
        if 0 < i1 < n1 && 0 < i2 < n2
			a1, a2 = x1 - i1, x2 - i2
		    b1, b2 = 1 - a1, 1 - a2 
            irreg[id] = regul[i1,i2]*b1*b2 + regul[i1+1,i2]*a1*b2 +
			            regul[i1,i2+1]*b1*a2 + regul[i1+1,i2+1]*a1*a2
        end
    end
    return irreg
end

# ╔═╡ a17f6efc-3d4e-4706-b386-df2d8e21e667
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
        if 0 < i1 < n[1] && 0 < i2 < n[2]
			a1, a2 = x1 - i1, x2 - i2
			b1, b2 = 1 - a1, 1 - a2 
			regul[i1,i2]     += irreg[id]*b1*b2
			regul[i1+1,i2]	 += irreg[id]*a1*b2
			regul[i1,i2+1]   += irreg[id]*b1*a2
			regul[i1+1,i2+1] += irreg[id]*a1*a2
        end
    end
    return regul
end

# ╔═╡ 02a2980c-26f7-4ed0-9eb0-5db2bb46d6c5
begin
	lat = -185:185 # latitude
	lon = -127:127 # longitude
	nlat, nlon = length(lat), length(lon)
end

# ╔═╡ 4cee4eeb-4a07-4734-9c2a-7e10f21576d4
ϵ = 0.01 # regularization parameter

# ╔═╡ 34cfabc4-2fc8-4753-82ce-f38b1ec34663
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

# ╔═╡ e2e62e28-7736-40af-9176-6877e30a185a
md"""
!!! assignment
    ## Task 2

    Find out experimentally how many iterations are needed for the conjugate gradient algorithm above to converge.
"""

# ╔═╡ abbc9694-136d-47bd-8728-a3dc04a0ba82
md"""
### Shaping regularization

Our second approach will use shaping regularization, defining model shaping as smoothing.
"""

# ╔═╡ ad1a94ac-cf05-4aed-b13e-6b05cba6befd
import LocalSignalAttributes

# ╔═╡ c12f2caf-fb8b-4638-b5c1-477b682d187f
radius = [15, 15] # smoothing radius in samples

# ╔═╡ 2bfb49dd-d2b2-436b-98e6-770b41eed68a
shaping(x) = LocalSignalAttributes.smooth(x, radius)

# ╔═╡ 35c3c081-b7af-4f74-acda-44178ae44c93
# smoothing is self-adjoint
dottest(shaping, shaping, rain0, rain0)

# ╔═╡ 8cfae86c-7565-46af-8ac8-dfa7c71ba4cc
md"""
!!! assignment
    ## Task 3

    Create an animation showing how the rainfall produced by shaping regularization changes with the number of iterations.

	To learn how to create animations in Julia, you can check the documentation as follows:

    ```julia
    ?@gif
    ```
"""

# ╔═╡ 1ea54cfb-038d-4802-b7c0-8fa24610cd60
myplot(k) = plot(sin, 0, k, legend=:none) # function definition

# ╔═╡ af53636c-83da-4491-8e0b-4c210d9881ec
@gif for k in 1:10 # create an animated gif using 10 frames
    myplot(k)
end

# ╔═╡ 990cdc30-99af-4bda-a862-b30a492ef67a
md"""
!!! assignment

	## Task 4

    We can use either accuracy or efficiency metrics to compare alternative numerical methods for solving the same problem. The [BenchmarkTools](https://juliaci.github.io/BenchmarkTools.jl/stable/) Julia package provides a convenient way of comparing efficiency. Use **@benchmark** or **@btime** functions from **BenchmarkTools** to compare the efficiency of gradient regularization versus shaping regularization.
"""

# ╔═╡ bc5bcd91-a8eb-4f1f-862e-c61754c54179
md"""
### Triangulation

Let us now try to be more creative with the backward operator used in shaping regularization. We will experiment with triangulated linear interpolation.
"""

# ╔═╡ 7bb0dbd3-2638-4591-82ba-9fd65069fbbe
# four corners of the map
corners = [[lat[1],lon[1]], [lat[1],lon[end]], 
	       [lat[end],lon[1]], [lat[end],lon[end]]];

# ╔═╡ e49dbef1-6dee-490c-b72c-8821c656a7ca
# weather station coordinates + corners
stations = vcat([obsdata[1:2,k] for k in 1:size(obsdata, 2)], corners);

# ╔═╡ 4df38169-ad22-4919-a30b-a791408f7d3a
tri_rain = triangulate(stations)

# ╔═╡ a39b6c86-38bf-4750-a068-598975a84d8d
function plot_tri(tri::Triangulation, title)
    plt = Plots.plot()
	points = DelaunayTriangulation.get_points(tri)
    for edge in DelaunayTriangulation.get_edges(tri)
        if edge[1] >= 1 && edge[2] >= 1 
            a, b = points[edge[1]], points[edge[2]]
            plot!(plt, [a[1], b[1]], [a[2], b[2]], 
                  label=:none, color=:blue)
        end
    end	
    x, y = [p[1] for p in points], [p[2] for p in points]
    scatter!(plt, x, y, aspect_ratio=:equal, color=:red, 
             label=:none, box=:none, title=title)
    return plt
end

# ╔═╡ 20025ba0-faa7-459e-9306-ff3b2b24c23d
plot_tri(tri_rain, "Trianguled Swiss Weather Stations")

# ╔═╡ f4b26ae9-b55d-4432-bdea-eb16a4a89cd4
# triangle area
T(A, B, C) = ((C[1] - A[1])*(C[2] - B[2]) - 
              (C[2] - A[2])*(C[1] - B[1]))/2

# ╔═╡ 169fa9f6-2bd0-4a12-a580-dd81695a18da
# linear interpolation using triangulation
function lint(tri::Triangulation, points, data, X)
	a, b, c = find_triangle(tri, X)
	A, B, C = points[a], points[b], points[c]
	return (data[a]*T(X,B,C) + data[b]*T(A,X,C) + data[c]*T(A,B,X))/T(A,B,C)
end	

# ╔═╡ 72d011f5-20ff-48a6-ab7c-61b35761eaea
begin
	forward(x) = lint(x, obsdata, o=[lat[1], lon[1]])
	adjoint(y) = lint_adjoint(y, obsdata, [nlat, nlon], o=[lat[1], lon[1]])
end

# ╔═╡ 1fe80e71-efac-468b-b488-d814eeadc308
begin
	x = [1,2,3,4]
	x0 = zeros(4)
	A = [1 1 1 0;
	     1 2 0 0;
	     1 3 1 0;
	     1 4 0 1;
	     1 5 1 1]
	B = A'*A
	d = B*x
	y = A*x
end

# ╔═╡ ff839a02-8ec5-43ca-9006-5d95698b7b67
@assert x ≈ conjgrad(x -> B*x, float(d), x0, 5)

# ╔═╡ 78a75724-84ca-4db1-8296-13776bae2a0e
@assert x ≈ conjgrad(x -> A*x, x -> A'*x, x -> x, float(y), x0, 5)

# ╔═╡ 418aee40-e03d-4fb7-aff7-40201a88fcc8
dottest(forward, adjoint, rain0, obsdata[3,:])

# ╔═╡ f758cfa1-772a-4558-bb78-355c2a88f227
map10 = conjgrad(forward, adjoint, gradient, obsdata[3,:], rain0, ϵ, 10);

# ╔═╡ a611fcf8-ef24-4d77-a926-35c806e873a9
heatmap(lat, lon, map10', title="Gradient Regularization (10 iterations)", cmap=:viridis)

# ╔═╡ 0aed61d9-7760-47fb-821c-1c45ff4b367c
map100 = conjgrad(forward, adjoint, gradient, obsdata[3,:], rain0, ϵ, 100);

# ╔═╡ 01d56647-c7e7-44c2-96e0-83f0bb3901b3
heatmap(lat, lon, map100', title="Gradient Regularization (100 iterations)", cmap=:viridis)

# ╔═╡ 4b8b6dbd-4cf6-4db3-ade0-736e73b4ec9a
maps = conjgrad(forward, adjoint, shaping, obsdata[3,:], rain0, 10, ϵ);

# ╔═╡ ef815292-27b0-4fb0-be60-33d84575ff37
@time shaping(maps);

# ╔═╡ 65df5863-18dc-482a-95cd-bba382655755
@btime shaping(maps);

# ╔═╡ 86cc41fe-82b8-4326-b15e-c942ea0d0b7b
heatmap(lat, lon, maps', title="Shaping Regularization (10 iterations)", cmap=:viridis)

# ╔═╡ 29ed07f8-d0e8-4abe-85f7-9243545133ec
predict = lint(map100, alldata, o=[lat[1], lon[1]]);

# ╔═╡ feedad1d-ea8b-402a-8aa9-750def84b593
plot_corr(predict, "Gradient Regularization")

# ╔═╡ a5e86a14-84cd-4e5b-9e18-3370fcbd57d9
predicts = lint(maps, alldata, o=[lat[1], lon[1]]);

# ╔═╡ b58d3dde-5d53-44ab-9bf0-473f301d0624
plot_corr(predicts, "Shaping Regularization")

# ╔═╡ ad18f0ea-3f1e-4e1c-8ca2-d010b03f0c79
function linmap(tri::Triangulation, points, data)
	n1, n2 = length(lat), length(lon)
	T = eltype(data)
	map = Array{T}(undef, n1, n2)
	# add four corners
	pad = vcat(data, zeros(T, 4))
	@inbounds for i in 1:n1, j in 1:n2
		point = (lat[i], lon[j])
		map[i, j] = lint(tri, points, pad, point)
	end
	return map
end

# ╔═╡ bd5e31b6-5edc-460d-8518-eae2ed596ae7
mapt = linmap(tri_rain, stations, obsdata[3,:]);

# ╔═╡ 295b4268-cc16-45c6-82c5-a2e7d0ee1674
heatmap(lat, lon, mapt', title="Triangulation", cmap=:viridis)

# ╔═╡ 94270dc8-241d-4766-a297-179161c68971
md"""
Triangulation efficiently solves the problem of creating a rainfall map that fits the input data. However, the map looks unrealistic because it is not smooth. To combine the advantages of triangulation with a smoothing constraint, we can use triangulated interpolation as the backward operator in shaping regularization.
"""

# ╔═╡ d3371c1a-2336-4bee-9a82-7225af6e2cd3
md"""
### Triangulation as the Backward Operator

To reconstruct a smooth map, we can apply shaping iterations.

$$\mathbf{m}_{n+1} = \mathbf{S}_m\left[\mathbf{B\,d} + (\mathbf{I - B\,F})\,\mathbf{m}_n\right]\;,$$

where $\mathbf{m}_n$ is the map at the $n$-th iteration, $\mathbf{d}$ is the measured rainfall, $\mathbf{F}$ is forward interpolation from a map to irregular locations, $\mathbf{B}$ is backward interpolation by triangulation, and $\mathbf{S}$ is smoothing.
"""

# ╔═╡ ed455ade-5096-4d0e-b491-b54b79292246
backward(y) = linmap(tri_rain, stations, y)

# ╔═╡ ebe091a3-8bd2-40f4-b5f4-8c1119d0aefb
function shape(forward::Function, backward::Function, 
	           shaping::Function, x0::Array, niter::Int)
	x = deepcopy(x0)
	for iter in 1:niter
		x = shaping(x0 + x - backward(forward(x)))
	end
	return x
end

# ╔═╡ ab19f4da-b581-4cdd-b9a8-6732b7293123
mapts = shape(forward, backward, shaping, mapt, 10);

# ╔═╡ 8e946ffe-e5d1-41a1-a0e0-6655862baec7
heatmap(lat, lon, mapts', title="Triangulation + Shaping", cmap=:viridis)

# ╔═╡ 911286af-2a00-46a0-975f-f046cfc1cbfe
predictts = lint(mapts, alldata, o=[lat[1], lon[1]]);

# ╔═╡ f67458b7-e448-4ce7-b8b1-6dd63a5c10e0
plot_corr(predictts, "Triangulation + Shaping")

# ╔═╡ 7a79c9da-51d2-47ed-ac48-3396112976f7
md"""
!!! assignment

    ## Task 5

    To improve the convergence of shaping iterations, we observe that the method requires inverting 

    $$\mathbf{m}_{\infty} = \left[\mathbf{I} + \mathbf{S}_m\,(\mathbf{B\,F - I})\right]^{-1}\,\mathbf{S}_m\,\mathbf{B\,d}\;.$$

    The inverted matrix is not symmetric. Therefore, the conjugate gradient algorithm is not applicable. However, we can use a different iterative algorithm, such as GMRES.

    * Saad, Y. and Schultz, M.H., 1986. [GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems](https://epubs.siam.org/doi/abs/10.1137/0907058). SIAM Journal on scientific and statistical computing, 7(3), pp.856-869.
"""

# ╔═╡ 035b6cd9-a24e-460b-8f7b-911be89e013c
@assert x ≈ gmres(Float32.(B), Float32.(B*x))

# ╔═╡ 7163eaec-0c81-42e1-91f4-1af99e3ee14d
@assert x ≈ gmres(LinearMap(x -> B*x, length(x)), Float32.(B*x), 
                  maxiter=5, verbose=true)

# ╔═╡ 7937d4b9-a70f-44a4-93cf-94144f57d8fb
reshape(x, (2,2))

# ╔═╡ e076e48b-ce42-4edb-b46f-d9a1aea7db58
function shaping_operator(x::AbstractVector)
	# wrap I + S(BF - I) operator for abstract vectors
	x2 = reshape(x, size(mapt)) # 1-D to 2-D
	y2 = x2 + shaping(backward(forward(x2)) - x2)
	return vec(y2) # 2-D to 1-D
end

# ╔═╡ d3c0ec35-2f48-4fac-8d69-4c7fa8e1ba1b
lin_operator = LinearMap{Float32}(shaping_operator, length(mapt));

# ╔═╡ 742dd279-5fd1-4247-9815-f342eaeda727
md"""
**Your task**: Implement shaping-regularized interpolation with GMRES and compare the result with that from simple iterations.
"""

# ╔═╡ 235493c0-5036-4fcb-86ea-b642aa4f604a
md"""
!!! assignment

    ## Bonus Task

    Participate in the Spatial Interpolation Contest. Implement a method that would provide a better interpolation of the missing values than either of the methods above. You can change any of the parameters in the existing functions or write your own function but you can use only the 100 original data points as input.
"""

# ╔═╡ bea6e20d-81a7-4096-989f-6394d0b6d5e5
md"""
## References

* Combettes, P. L., and V. R. Wajs, 2005, [Signal recovery by proximal forward-backward splitting:](https://epubs.siam.org/doi/abs/10.1137/050626090) Multiscale Modeling and Simulation, 4, 1168–1200.
* Fomel, S., 2007, [Shaping regularization in geophysical estimation problems](https://library.seg.org/doi/full/10.1190/1.2433716): Geophysics, 72, R29–R36.
* Fomel, S., 2008, [Nonlinear shaping regularization in geophysical inverse problems](https://library.seg.org/doi/abs/10.1190/1.3059294): 78th Annual International Meeting, SEG, Expanded Abstracts, 2046–2051.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DelaunayTriangulation = "927a84f5-c5f4-47a5-9785-b46e178433df"
GMT = "5752ebe1-31b9-557e-87aa-f909b540aa54"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LinearMaps = "7a12625a-238d-50fd-b39a-03d52299707e"
LocalSignalAttributes = "0a92bf9b-4da3-44e8-9286-830175b27cf8"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUIExtra = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
BenchmarkTools = "~1.6.0"
DelaunayTriangulation = "~1.6.4"
GMT = "~1.22.2"
HTTP = "~1.11.0"
IterativeSolvers = "~0.9.4"
LinearMaps = "~3.11.3"
LocalSignalAttributes = "~1.0.3"
Plots = "~1.41.1"
PlutoUIExtra = "~0.1.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "e89bc53accf44748c4a18f3d56cbd6a8a7750c5c"

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

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Arrow_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Lz4_jll", "Thrift_jll", "Zlib_jll", "boost_jll", "snappy_jll"]
git-tree-sha1 = "2f57dbca4a69845200e9c9085b3968769d4e30d1"
uuid = "8ce61222-c28f-5041-a97a-c2198fb817bf"
version = "10.0.1+0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "535c80f1c0847a4c967ea945fca21becc9de1522"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.7+0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

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
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

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

[[deps.DataPipes]]
git-tree-sha1 = "29077a8d5c093f4e0988e92c0d76f56c4c581900"
uuid = "02685ad9-2d12-40c3-9f73-c6aeda6a7ff5"
version = "0.3.18"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6c72198e6a101cccdd4c9731d3985e904ba26037"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.1"

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

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "5620ff4ee0084a6ab7097a27ba0c19290200b037"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.4"

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

[[deps.EnumX]]
git-tree-sha1 = "bddad79635af6aec424f53ed8aad5d7555dc6f00"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.5"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7bb1361afdb33c7f2b085aa49ea8fe1b0fb14e58"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.1+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "83dc665d0312b41367b7263e8a4d172eac1897f4"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.4"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3a948313e7a41eb1db7a1e733e6335f17b4ab3c4"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "7.1.1+0"

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

[[deps.GDAL_jll]]
deps = ["Arrow_jll", "Artifacts", "Expat_jll", "GEOS_jll", "HDF5_jll", "JLLWrappers", "LibCURL_jll", "LibPQ_jll", "Libdl", "Libtiff_jll", "NetCDF_jll", "OpenJpeg_jll", "PROJ_jll", "SQLite_jll", "Zlib_jll", "Zstd_jll", "libgeotiff_jll"]
git-tree-sha1 = "1740251f4f1ce850d4eaad052d5e419229d4b2af"
uuid = "a7073274-a066-55f0-b90d-d619367d196c"
version = "301.900.0+0"

[[deps.GEOS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "79f6dfc0bd6f5d46b93b5938de4530c9d1e7de34"
uuid = "d604d12d-fa86-5845-992e-78dc15976526"
version = "3.14.0+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GMT]]
deps = ["Dates", "Downloads", "GDAL_jll", "GMT_jll", "Ghostscript_jll", "LASzip_jll", "LinearAlgebra", "PROJ_jll", "PrecompileTools", "PrettyTables", "Printf", "Statistics", "Tables"]
git-tree-sha1 = "72e25c5064945ddb5f9aedffc544191057d44630"
uuid = "5752ebe1-31b9-557e-87aa-f909b540aa54"
version = "1.22.2"

    [deps.GMT.extensions]
    GMTDataFramesExt = "DataFrames"

    [deps.GMT.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"

[[deps.GMT_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "FFTW_jll", "GDAL_jll", "Ghostscript_jll", "Glib_jll", "JLLWrappers", "LAPACK32_jll", "LLVMOpenMP_jll", "LibCURL_jll", "Libdl", "NetCDF_jll", "OpenBLAS32_jll", "PCRE_jll", "PROJ_jll"]
git-tree-sha1 = "5bac9b9bd1af9bef38bb686d7d460c54bc19797d"
uuid = "b68b8c3f-ed99-5bef-9675-4739d9426b26"
version = "6.5.2+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "629693584cef594c3f6f99e76e7a7ad17e60e8d5"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.7"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a8863b69c2a0859f2c2c87ebdc4c6712e88bdf0d"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.7+0"

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
git-tree-sha1 = "50c11ffab2a3d50192a228c313f05b5b5dc5acb2"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.0+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "82a471768b513dc39e471540fdadc84ff80ff997"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.3+3"

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

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "3d468106a05408f9f7b6f161d9e7715159af247b"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.12.2+0"

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

[[deps.ICU_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b3d8be712fbf9237935bde0ce9b5a736ae38fc34"
uuid = "a51ab1cf-af8e-5615-a023-bc2c838bba6b"
version = "76.2.0+0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Random", "RoundingEmulator"]
git-tree-sha1 = "79342df41c3c24664e5bf29395cfdf2f2a599412"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.36"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "5fbb102dcb8b1a858111ae81d56682376130517d"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.11"
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

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "59545b0a2b27208b0650df0a46b8e3019f85055b"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.4"

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
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.Kerberos_krb5_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0f2899fdadaab4b8f57db558ba21bdb4fb52f1f0"
uuid = "b39eb1a6-c29a-53d7-8c32-632cd16f18da"
version = "1.21.3+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LAPACK32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "libblastrampoline_jll"]
git-tree-sha1 = "500a74699be04564437ad7944f49ea5179b5bb4e"
uuid = "17f450c3-bd24-55df-bb84-8c51b4b939e3"
version = "3.12.1+0"

[[deps.LASzip_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6a5de2baf47bf8ada29ad6f566d8ce1a50dc5cfe"
uuid = "8372b9c3-1e34-5cc3-bfab-1a98e101de11"
version = "3.4.3000+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

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

[[deps.LibPQ_jll]]
deps = ["Artifacts", "ICU_jll", "JLLWrappers", "Kerberos_krb5_jll", "Libdl", "OpenSSL_jll", "Zstd_jll"]
git-tree-sha1 = "7757f54f007cc0eb516a5000fb9a6fc19a49da7e"
uuid = "08be9ffa-1c94-5ee5-a977-46a84ec9b350"
version = "16.8.0+0"

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
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2a7a12fc0a4e7fb773450d17975322aa77142106"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.2+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ee79c3208e55786de58f8dcccca098ced79f743f"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.3"

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

    [deps.LinearMaps.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "fa7fd067dca76cadd880f1ca937b4f387975a9f5"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.16.0+0"

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
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "191686b1ac1ea9c89fc52e996ad15d1d241d1e33"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.10.1+0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "d72d0ecc3f76998aac04e446547259b9ae4c265f"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.3.1+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "e214f2a20bdd64c04cd3e4ff62d3c9be7e969a59"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.4+0"

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
git-tree-sha1 = "3cce3511ca2c6f87b19c34ffc623417ed2798cbd"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.10+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

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

[[deps.NetCDF_jll]]
deps = ["Artifacts", "Blosc_jll", "Bzip2_jll", "HDF5_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "TOML", "XML2_jll", "Zlib_jll", "Zstd_jll", "libzip_jll"]
git-tree-sha1 = "4686378c4ae1d1948cfbe46c002a11a4265dcb07"
uuid = "7243133f-43d8-5620-bbf4-c2c921802cf3"
version = "400.902.211+1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ece4587683695fe4c5f20e990da0ed7e83c351e7"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.29+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "567515ca155d0020a45b05175449b499c63e7015"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.29+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "libpng_jll"]
git-tree-sha1 = "7dc7028a10d1408e9103c0a77da19fdedce4de6c"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.5.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "21cd48e9a91fd3dfaac23701e4e338e6bca4c209"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "4.1.8+1"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

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

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ccf0e9339e1f3e66e241ce01bbcbf57a0a9c15a1"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.45.0+0"

[[deps.PROJ_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "Libtiff_jll", "SQLite_jll"]
git-tree-sha1 = "84aa844bd56f62282116b413fbefb45e370e54d6"
uuid = "58948b4f-47e0-5654-a9ad-f609743f8632"
version = "901.300.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1f7f9bbd5f7a2e5a9f7d96e51c9754454ea7f60b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.4+0"

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
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "12ce661880f8e309569074a61d3767e5756a199f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.1"

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
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "8329a3a4f75e178c11c1ce2342778bcbbbfa7e3c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.71"

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
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
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

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SQLite_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "9a325057cdb9b066f1f96dc77218df60fe3007cb"
uuid = "76ed43ae-9a5d-5a62-8c75-30186b810ce8"
version = "3.48.0+0"

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
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

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
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2c962245732371acd51700dbb268af311bddd719"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.6"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

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

[[deps.Thrift_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "boost_jll"]
git-tree-sha1 = "fd7da49fae680c18aa59f421f0ba468e658a2d7a"
uuid = "e0b8ae26-5307-5830-91fd-398402328850"
version = "0.16.0+0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

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

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

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

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "4909eb8f1cbf6bd4b1c30dd18b2ead9019ef2fad"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.18.1+0"

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

[[deps.boost_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7a89efe0137720ca82f99e8daa526d23120d0d37"
uuid = "28df3c45-c428-5900-9ff8-a3135698ca75"
version = "1.76.0+1"

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

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1aa23f01927b2dac46db77a56b31088feee0a491"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.4+0"

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

[[deps.libgeotiff_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "Libtiff_jll", "PROJ_jll"]
git-tree-sha1 = "c48ca6e850d4190dcb8e0ccd220380c2bc678403"
uuid = "06c338fa-64ff-565b-ac2f-249532af990e"
version = "100.701.300+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libzip_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "OpenSSL_jll", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "86addc139bca85fdf9e7741e10977c45785727b7"
uuid = "337d8026-41b4-5cde-a456-74a10e5b31d1"
version = "1.11.3+0"

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
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.snappy_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ca88363dd41d2547f52118287dd34dbbc14f3eb7"
uuid = "fe1e1685-f7be-5f59-ac9f-4ca204017dfd"
version = "1.2.3+0"

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
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
"""

# ╔═╡ Cell order:
# ╟─7e40faf0-a503-4c98-a9d2-9a824a594768
# ╟─cdc06c3e-c3ae-491a-b974-f936c95af886
# ╟─613746d8-8817-4160-bb29-b3ceaa0cd0c2
# ╟─5466a099-b57c-4a7c-aea0-91a258d834d4
# ╟─20d7074b-7daa-420f-81a7-7bc63fa10a1c
# ╟─c3570349-fdee-490f-b1d6-0f13daf82a8d
# ╟─e8a121b1-fff3-405c-989d-f6969cc5da6e
# ╟─028eda0f-4abb-42f0-891a-a0e646982814
# ╟─504b099f-8bb9-455e-8e5a-2300bd6de853
# ╟─43ad1b2b-1fc1-48af-881f-c1daf33e47be
# ╠═adaaeb9e-2cef-4667-bf78-3854bf9271e2
# ╠═7168dd81-1f22-412a-954e-3f9c5c3f050c
# ╠═1fe80e71-efac-468b-b488-d814eeadc308
# ╠═ff839a02-8ec5-43ca-9006-5d95698b7b67
# ╟─95b3b4f3-c9dd-41ff-a8f6-dd33b39d2018
# ╟─18ac4fcb-6e6e-40bc-8e58-07a766ede06d
# ╠═41ebe5b1-15da-4f88-ae91-f9c412d15ca5
# ╠═78a75724-84ca-4db1-8296-13776bae2a0e
# ╟─b89a90b1-643c-4b4d-9c26-cd1779ea8c78
# ╟─a050e60a-d63e-4ced-9a58-70e7749e2e86
# ╠═fef2bc4a-c36d-47be-a00b-2e0c42776e3b
# ╠═9dec3276-49d9-4d15-82e8-ff79eb38e53a
# ╟─0e518bda-c96b-41ca-a98a-5acb1f0d98e0
# ╟─5c0c26ec-995c-4c8a-80a9-5668af411544
# ╠═28558f29-1a20-4408-8c85-cf82dd7db05d
# ╠═5e349cf8-deee-4490-a9c9-340c7ef75854
# ╠═aae2ac2d-80da-4cd3-ae0c-d49eff6a810f
# ╠═0f28e9f6-5670-46bc-a1a5-11cb6dbbad39
# ╠═81e89634-87c1-4251-a1fd-89d5a02f8f79
# ╟─a2f3c5ad-9128-44b7-8f2d-919c3742960d
# ╟─c4fa6719-72cf-49c1-bd65-fc44e15af8b0
# ╠═643afb4d-2734-4037-91a9-14a6be815ba8
# ╠═a662f040-1375-46c4-b9fc-0998dc6d357b
# ╠═7ac29fda-34b0-4f6c-8725-9a4bd1a2a09b
# ╠═dd0f8427-f880-4264-8ef3-76b7e9ab593e
# ╠═59795e0f-fc90-4575-8ce1-8affe855096f
# ╠═e57684e9-d6a1-42f5-a017-4954dcb4ddfc
# ╠═6a623beb-3dab-4172-9064-81025d599352
# ╠═ac516e49-c7d0-4bf7-a0df-67cf25aec5ec
# ╠═7d0e3bbb-8bbc-4502-825c-8c74b09c4f7e
# ╠═991d72c3-2843-4409-b259-c3044f7c0934
# ╠═a17f6efc-3d4e-4706-b386-df2d8e21e667
# ╠═02a2980c-26f7-4ed0-9eb0-5db2bb46d6c5
# ╠═72d011f5-20ff-48a6-ab7c-61b35761eaea
# ╠═418aee40-e03d-4fb7-aff7-40201a88fcc8
# ╠═4cee4eeb-4a07-4734-9c2a-7e10f21576d4
# ╠═f758cfa1-772a-4558-bb78-355c2a88f227
# ╠═a611fcf8-ef24-4d77-a926-35c806e873a9
# ╠═0aed61d9-7760-47fb-821c-1c45ff4b367c
# ╠═01d56647-c7e7-44c2-96e0-83f0bb3901b3
# ╠═29ed07f8-d0e8-4abe-85f7-9243545133ec
# ╠═34cfabc4-2fc8-4753-82ce-f38b1ec34663
# ╠═feedad1d-ea8b-402a-8aa9-750def84b593
# ╟─e2e62e28-7736-40af-9176-6877e30a185a
# ╟─abbc9694-136d-47bd-8728-a3dc04a0ba82
# ╠═ad1a94ac-cf05-4aed-b13e-6b05cba6befd
# ╠═c12f2caf-fb8b-4638-b5c1-477b682d187f
# ╠═2bfb49dd-d2b2-436b-98e6-770b41eed68a
# ╠═35c3c081-b7af-4f74-acda-44178ae44c93
# ╠═4b8b6dbd-4cf6-4db3-ade0-736e73b4ec9a
# ╠═86cc41fe-82b8-4326-b15e-c942ea0d0b7b
# ╠═a5e86a14-84cd-4e5b-9e18-3370fcbd57d9
# ╠═b58d3dde-5d53-44ab-9bf0-473f301d0624
# ╟─8cfae86c-7565-46af-8ac8-dfa7c71ba4cc
# ╠═1ea54cfb-038d-4802-b7c0-8fa24610cd60
# ╠═af53636c-83da-4491-8e0b-4c210d9881ec
# ╟─990cdc30-99af-4bda-a862-b30a492ef67a
# ╠═1d477149-313a-4265-b498-09ba3c0eafd4
# ╠═ef815292-27b0-4fb0-be60-33d84575ff37
# ╠═65df5863-18dc-482a-95cd-bba382655755
# ╟─bc5bcd91-a8eb-4f1f-862e-c61754c54179
# ╠═7bb0dbd3-2638-4591-82ba-9fd65069fbbe
# ╠═e49dbef1-6dee-490c-b72c-8821c656a7ca
# ╠═8164a2e1-467a-4a4b-9240-f28da7df2b40
# ╠═4df38169-ad22-4919-a30b-a791408f7d3a
# ╠═a39b6c86-38bf-4750-a068-598975a84d8d
# ╠═20025ba0-faa7-459e-9306-ff3b2b24c23d
# ╠═f4b26ae9-b55d-4432-bdea-eb16a4a89cd4
# ╠═169fa9f6-2bd0-4a12-a580-dd81695a18da
# ╠═ad18f0ea-3f1e-4e1c-8ca2-d010b03f0c79
# ╠═bd5e31b6-5edc-460d-8518-eae2ed596ae7
# ╠═295b4268-cc16-45c6-82c5-a2e7d0ee1674
# ╟─94270dc8-241d-4766-a297-179161c68971
# ╟─d3371c1a-2336-4bee-9a82-7225af6e2cd3
# ╠═ed455ade-5096-4d0e-b491-b54b79292246
# ╠═ebe091a3-8bd2-40f4-b5f4-8c1119d0aefb
# ╠═ab19f4da-b581-4cdd-b9a8-6732b7293123
# ╠═8e946ffe-e5d1-41a1-a0e0-6655862baec7
# ╠═911286af-2a00-46a0-975f-f046cfc1cbfe
# ╠═f67458b7-e448-4ce7-b8b1-6dd63a5c10e0
# ╟─7a79c9da-51d2-47ed-ac48-3396112976f7
# ╠═5f3e2b21-2a7b-49b1-af1e-01272da47c58
# ╠═035b6cd9-a24e-460b-8f7b-911be89e013c
# ╠═7163eaec-0c81-42e1-91f4-1af99e3ee14d
# ╠═7937d4b9-a70f-44a4-93cf-94144f57d8fb
# ╠═e076e48b-ce42-4edb-b46f-d9a1aea7db58
# ╠═d3c0ec35-2f48-4fac-8d69-4c7fa8e1ba1b
# ╟─742dd279-5fd1-4247-9815-f342eaeda727
# ╟─235493c0-5036-4fcb-86ea-b642aa4f604a
# ╟─bea6e20d-81a7-4096-989f-6394d0b6d5e5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
