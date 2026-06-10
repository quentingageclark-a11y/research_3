### A Pluto.jl notebook ###
# v0.20.27

#> [frontmatter]
#> chapter = 4
#> section = 26
#> order = 26
#> image = "https://raw.githubusercontent.com/fonsp/Pluto.jl/580ab811f13d565cc81ebfa70ed36c84b125f55d/demo/plutodemo.gif"
#> title = "26 Data Analysis Notebook"
#> tags = ["module4", "track_julia", "track_material", "Pluto", "PlutoUI"]
#> layout = "layout.jlhtml"

using Markdown
using InteractiveUtils

# ╔═╡ 64bd1fbc-9a20-4902-afda-b4f6d04c8094
using Plots

# ╔═╡ 36dd5738-1419-48e4-859b-532d3638a658
using FastMarching

# ╔═╡ 0ce44c53-7be7-4c58-8743-c8ec8424aa74
using LinearAlgebra

# ╔═╡ 17ee38ba-49bf-4083-9b67-2f04f2d0e8f3
using Random

# ╔═╡ e88510fb-4681-4bf7-bcf1-7cfad17a2559
import HTTP

# ╔═╡ c86323f6-9cc4-434d-b147-f4107fa61a39
let
    notebook_dir = @__DIR__
    toc_path = joinpath(notebook_dir, "TOC_Notebook.jl")
    
    Markdown.parse("[⬅️ Back to Table of Contents](./open?path=$(HTTP.escapeuri(toc_path)))")
end

# ╔═╡ 11102582-1e4f-11f0-2558-65ff8aa2a1f8
md"""
# Adaptive regularization

We solved nonstationary problems using stationary regularization, such as smoothing with a fixed radius. This chapter will investigate different ways to adapt regularization to the problem.
"""

# ╔═╡ a238973b-e8f4-40c0-9d60-7194d4871438
md"""
## Nonstationary smoothing

How to implement non-stationary smoothing? The analysis of the Taylor expansion around the zero frequency suggests achieving it by interpolating between two triangles as follows:

$$T_R(Z) = \displaystyle \frac{(N+1)^2-R^2}{2\,N+1}\,T_N(Z) + \frac{R^2-N^2}{2\,N+1}\,T_{N+1}(Z)\;,$$

where $N$ is the largest integer smaller than $R$ so that $N \le R \le N+1$.
"""

# ╔═╡ 8e3b79dd-0af4-4710-b2e1-e3ac1aab1ac0
function doubint!(x::Vector) 
    "causal and anticausal integration in place"
    n = length(x)  
    for range in (1:n, n:-1:1)
        t = zero(eltype(x))
        for i in range
            t += x[i]
            x[i] = t
        end
    end
end

# ╔═╡ 96c62806-16a4-4b69-a02e-5a7d42ccd065
function fold(t::Vector, nb::Integer)
    "reflecting boundary conditions for smoothing"
    nt = length(t)
    n = nt - 2*nb
    # copy middle
    x = t[1+nb:n+nb]
    # reflections from the right side 
    for j in nb+n:2*n:nt
        for i in 1:min(n,nt-j); x[n+1-i] += t[j+i]; end
        for i in 1:min(n,nt-j-n); x[i] += t[j+n+i]; end
    end
    # reflections from the left side
    for j in nb+1:-2*n:1
        for i in 1:min(n,j-1); x[i] += t[j-i]; end
        for i in 1:min(n,j-1-n); x[n+1-i] += t[j-i-n]; end
    end
    return x
end

# ╔═╡ e6b8f7b9-29b5-48be-b8c1-e0105986eb56
function fold_adj(x::Vector, nb::Integer)
    "reflecting boundary conditions for smoothing"
	n = length(x)
    nt = n + 2*nb
	t = zeros(eltype(x), nt)
    # copy middle
    t[1+nb:n+nb] = x
    # reflections from the right side 
    for j in nb+n:2*n:nt
        for i in 1:min(n,nt-j); t[j+i] += x[n+1-i]; end
        for i in 1:min(n,nt-j-n); t[j+n+i] += x[i]; end
    end
    # reflections from the left side
    for j in nb+1:-2*n:1
        for i in 1:min(n,j-1);  t[j-i] += x[i]; end
        for i in 1:min(n,j-1-n); t[j-i-n] += x[n+1-i]; end
    end
    return t
end

# ╔═╡ 44f87835-008d-4670-84d7-94940895a565
function smoothr(x::Vector, r::Vector{T}) where T <: Real
    "smoothing by triangle filtering with reflecting boundaries"
    n = length(x)
	y = similar(x)
	nb = ceil(Int, maximum(r)) + 1
    t = fold_adj(x, nb)
	doubint!(t)
    for i in 1:n
		ri = r[i]
		nt = floor(Int, ri)
		nt1 = nt+1
		wt  = (nt1*nt1 - ri*ri)/(nt*nt * (nt+nt1))
		wt1 = (ri*ri  -  nt*nt)/(nt1*nt1*(nt+nt1))
		y[i] = 2*(wt+wt1)*t[i+nb] - 
	    		(t[i+nb-nt1] + t[i+nb+nt1])*wt1 - 
	    		(t[i+nb-nt]  + t[i+nb+nt ])*wt
    end
	return y
end

# ╔═╡ b69f07b7-6857-46f4-8dd6-0afc720c0a5e
function stems(data, label, color) 
    plt=plot(zeros(Float32, 13), label=:none, color=:black)
    clip = 1.1*maximum(abs.(data))
    plot!(plt, data, line=:stem, marker=:circle, 
          label=label, color=color, legend=:outerleft, 
          xlim=[0.5, 13.5], ylim=[-clip, clip], border=:none)   
    return plt
end

# ╔═╡ 231316b2-3d2c-4898-a15b-9262cac03294
begin
	spike = zeros(13)
	spike[7] = 1
end

# ╔═╡ e856d9de-5b21-4276-bd7d-8c8cab684827
begin
	ps = Array{Plots.Plot}(undef, 6)
	rs = range(start=3, stop=4, length=6)
	r = similar(spike)
	for k in 1:6
		r .= rs[k]
		triangle = smoothr(spike, r)
		ps[k] = stems(triangle, "radius=$(rs[k])", k)
	end
	p1 = plot(ps[1], ps[2], ps[3], layout=(3, 1))
	p2 = plot(ps[4], ps[5], ps[6], layout=(3, 1))
	plot(p1, p2, layout=(1, 2))
end

# ╔═╡ 0abe931a-737f-408a-bdec-7a41bf014a2b
function smooth2(x::Array, r::Array) 
	"smoothing in 2D"
	n1, n2 = size(x)
    y = deepcopy(x)
	for i2 in 1:n2
		y[:,i2] = smoothr(y[:,i2], r[:,i2,1])
	end
	for i1 in 1:n1
		y[i1,:] = smoothr(y[i1,:], r[i1,:,2])
	end
   return y
end

# ╔═╡ 41db06e2-a9aa-4d6d-a378-d01d8f5e2a13
md"""
### Using non-stationary smoothing in data reconstruction

We will return to the rainfall data interpolation example to demonstrate how to achieve fast data gridding with non-stationary smoothing.
"""

# ╔═╡ a5e515ab-5a1b-46a6-8288-f5ea3a5bfde8
begin
	# download data files
	download("https://ahay.org/data/rain/alldata.rsf@","alldata.bin")
	download("https://ahay.org/data/rain/obsdata.rsf@","obsdata.bin")
end

# ╔═╡ ce5a04e4-f8b0-487d-a3af-6f4f720e4570
begin
	# read data
	alldata = Array{Float32}(undef, 3, 467); 
	obsdata = Array{Float32}(undef, 3, 100); 
	read!("alldata.bin", alldata)
	read!("obsdata.bin", obsdata)
end

# ╔═╡ 48a90528-5527-4c17-8a69-d86c07e1314f
function compute_distance(source::Array, nx::Int, ny::Int; x0=0, y0=0, dx=1, dy=1)
	# unit velocity
	velocity = ones(eltype(source), nx, ny)
	xy = similar(source[1:2,:])
	xy[1,:] = (source[1,:] .- x0)/dx .+ 1
	xy[2,:] = (source[2,:] .- y0)/dy .+ 1
	return FastMarching.msfm(velocity, xy, true, true)
end

# ╔═╡ 3f828447-5a33-42ab-a888-7c8480336e72
begin
	lat = -185:185 # latitude
	lon = -127:127 # longitude
	nx, ny = length(lat), length(lon)
	x0, y0 = lat[1], lon[1]
end

# ╔═╡ 7244a1da-8ba8-4546-8089-eb8561af3471
dist = compute_distance(obsdata, nx, ny, x0=x0, y0=y0);

# ╔═╡ 547a6bc6-9c75-4c16-9c66-4c8844a1d91f
heatmap(lat, lon, dist', title="Distance", cmap=:coolwarm)

# ╔═╡ e1d2dced-b59b-4bed-a64b-b005f03c9397
function bin(sources::Array, values::Array, nx::Int, ny::Int; x0=0, y0=0, dx=1, dy=1)
	# bin initial data to the grid
	binned = zeros(eltype(sources), nx, ny)
	for n in 1:size(sources,2)
    	x, y = sources[1,n], sources[2,n]
    	i, j = round(Int, 1 + (x - x0)/dx), round(Int, 1 + (y - y0)/dy)
    	if 1 <= i <= nx && 1 <= j <= ny
        	binned[i,j] = values[n]
    	end
	end
	return binned
end

# ╔═╡ e73dbb85-d21b-47bf-b199-1a15c515b95c
binned = bin(obsdata, obsdata[3,:], nx, ny, x0=x0, y0=y0);

# ╔═╡ 4537b2d6-90c7-48c0-9a29-eb592c699fad
function nearest_neighbor(distance)
	# sort grid indices by distance
	grid = CartesianIndices(distance)[:];
	sort!(grid, by=i -> distance[i]);
	neighbor = similar(grid)
	d1, d2 = CartesianIndex((1,0)), CartesianIndex((0,1))
	for i in 1:length(grid)
		g = grid[i]
    	neighbors = filter(x -> x in grid, [g, g+d1, g-d1, g+d2, g-d2])
    	d, j = findmin(distance[neighbors])
    	neighbor[i] = neighbors[j]
	end
	return hcat(grid, neighbor)
end

# ╔═╡ b3e3c1e5-cb8c-4e4b-b49d-a25084de0081
grid = nearest_neighbor(dist)

# ╔═╡ d219e674-320f-46d7-a3b5-b122c2be77a0
function interpolate_nearest(binned, grid)
	nearest = copy(binned)
	for (i, j) in eachrow(grid)
    	nearest[i] = nearest[j]
	end
	return nearest
end

# ╔═╡ a1f920ad-c494-424b-96ea-2d886d05bee9
rain = interpolate_nearest(binned, grid);

# ╔═╡ d025636b-8b39-4441-a7e8-c540c4a332db
heatmap(lat, lon, rain', title="Rainfall Interpolation", cmap=:viridis)

# ╔═╡ 2fe180bb-ffa7-4f25-b162-7a946c3e83e2
md"""
Starting with nearest-neighbor interpolation, we will perform smoothing only once, using a smoothing radius equal to the distance to the nearest weather station.
"""

# ╔═╡ 36308986-5642-436f-885c-5fb7b33ed0f4
rad = dist .+ 1; # the smallest smoothest radius is 1

# ╔═╡ b75a4dd3-cf6a-4bf7-869f-60de08ad492c
smoothed = smooth2(rain, cat(rad, rad; dims=3));

# ╔═╡ d78fffcd-9d17-424d-887b-fa3cc448e570
heatmap(lat, lon, smoothed', title="Smoothed Rainfall Interpolation", cmap=:viridis)

# ╔═╡ d2bd20d0-d665-4ae1-b9ab-99596317b9af
function lint(regul, coord; d=[1,1], o=[0,0])
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

# ╔═╡ 6cd9e01c-4542-4ea7-a128-6856a0a4a163
begin
	exact = alldata[3,:]
	predict = lint(smoothed, alldata; o=[lat[1], lon[1]]);
end

# ╔═╡ b649d55a-4def-4c98-bd21-19be81dc3f62
begin
	lim = [-10, 600]
	cc = (exact ⋅ predict)/(sqrt(exact ⋅ exact) * sqrt(predict ⋅ predict))
	scatter(exact, predict, xlabel="True", ylabel="Predicted", 
		title="Smoothed Interpolation, cc=$(Float16(cc))",
	    aspect_ratio=:equal, xlim=lim, ylim=lim, label=:none)
	plot!(lim, lim, label=:none)
end

# ╔═╡ 62682722-5fea-4449-a7d4-1f8973515ee4
md"""
### Estimating triangle radius

Suppose we have two datasets $\mathbf{d}_1$ and $\mathbf{d}_2$ connected via triangle smoothing. How can we determine the smoothing radius?

The relationship is

$$\mathbf{T}(\mathbf{r})\,\mathbf{d}_1 \approx \mathbf{d}_2\;,$$

where the dependence of triangle smoothing $\mathbf{T}$ on radius $\mathbf{r}$ is nonlinear. The Gauss-Newton approach calls for iterative estimation through linearization

$$\left[\mathbf{T}(\mathbf{r}_n) + \mathbf{T}'(\mathbf{r}_n)\,(\mathbf{r}_{n+1} - \mathbf{r}_n)\right]\,\mathbf{d}_1 \approx \mathbf{d}_2\;,$$
"""

# ╔═╡ 359c5695-45bb-4588-9f60-e5df5d394bb5
md"""
Writing the frequency-domain representation of the triangle filter 

$$T(R,\omega) = \displaystyle \frac{1}{R^2}\,\left[\frac{\sin\left(\frac{R\,\omega\Delta t}{2}\right)}{\sin\left(\frac{\omega\Delta t}{2}\right)}\right]^2\;,$$

we can differentiate it with respect to $R$ to find the derivative representation, as follows:

$$\displaystyle \frac{\partial T}{\partial R}(R,\omega) = i\omega \left[\frac{-i \Delta t \sin\left(\frac{R\,\omega \Delta t}{2}\right)}{2R^2\sin^2\left(\frac{\omega \Delta t}{2}\right)}\right] - \frac{2}{R}\,T(R,\omega).$$
"""

# ╔═╡ 50f6fcaa-10e4-41bc-aec3-5fd86b5f0775
md"""
To return to the time domain, we recognize that the $i\omega$ term represents the derivative and that, for integer $R=N$, the expression in the square brackets can be implemented as a digital filter

$$\displaystyle \frac{Z^N-Z^{-N}}{N^2(1-Z)(1-Z^{-1})}\;.$$

For non-integer $R$, the term is computed via interpolation.
"""

# ╔═╡ 516fe177-9a8a-406b-99c7-245449b7066c
md"""
* S. Greer and S. Fomel, 2018, Matching and merging high-resolution and legacy seismic images: Geophysics, v. 83, V115-V122.
* R. Alomar and S. Fomel, 2022. Least-squares non-stationary triangle smoothing: Second International Meeting for Applied Geoscience & Energy, 2847-2851.
"""

# ╔═╡ 63f60cf5-8e11-41be-ba30-fbf357b8eb87
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

# ╔═╡ ebf4263e-5f8c-42f0-bedf-7d5e163a1d85
function smooth_der(x::Vector, r::Vector{T}) where T <: Real
    "derivative smoothing by triangle filtering"
    n = length(x)
	y = similar(x)
	nb = ceil(Int, maximum(r)) + 1
    t = fold_adj(x, nb)
	doubint!(t)
    for i in 1:n
		ri = r[i]
		nt = floor(Int, ri)
		nt1 = nt+1
		wt = (nt1-ri)/(nt*nt)
		wt1 = (ri-nt)/(nt1*nt1)
		y[i] = (t[i+nb-nt]  - t[i+nb+nt ])*wt - 
		       (t[i+nb-nt1] - t[i+nb+nt1])*wt1 
    end
	return deriv(y) - 2*smoothr(x,r) ./ r
end

# ╔═╡ 020ea553-a7a5-457c-9db5-19859fca4592
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

# ╔═╡ a3100bd3-f0ea-4db1-a128-5ee8ee99b4cc
begin
	Random.seed!(2025)
	data1 = randn(51)
	butterworth!(true, data1, 0.2, 6)
	n = -25:25
	radius = 3*exp.(-0.025*(n .^2)) .+ 1
	radius0 = ones(51)
	data2 = smoothr(data1, radius)
end

# ╔═╡ 6ad926d8-d80a-46b4-a483-ebc5c5caa8c6
function conjgrad(forward::Function, adjoint::Function, shaping::Function, ϵ::Float64, d::Array, p0::Array, niter::Int; tol=1.0e-6, verb=true)
    "Conjugate-gradient algorithm for shaping regularization"
    p = deepcopy(p0)
    x = shaping(p)
    r = forward(x) .- d      
    sp, sx, sr = similar(p), similar(x), similar(r)
    g0 = gnp = zero(Float64)
    for iter in 1:niter
        gx = adjoint(r) - ϵ*x
        gp = shaping(gx) + ϵ*p
        gx = shaping(gp)
        gr = forward(gx)
        
        gn = real(gp ⋅ gp)
		if verb
	        @show iter, gn 
		end
        
        if iter==1
            sp, sx, sr = gp, gx, gr
			g0 = gn
        else
            β = gn/gnp
			γ = gn/g0
			if (β < tol || gn/g0 < tol)
				if verb
					println("converged: β=$(β) γ=$(γ)")
				end
				break
			end
            sp = gp + β*sp
            sx = gx + β*sx
            sr = gr + β*sr
        end
        gnp = gn
        
        α = real(sr ⋅ sr + ϵ*(sp ⋅ sp -  sx ⋅ sx))
        α = - gn/α
        
        p = p + α*sp
        x = x + α*sx
        r = r + α*sr
    end
    return x
end

# ╔═╡ 2e3e178a-bc18-46e2-b118-4634987a8a1a
function smooth(x::Vector, nb::Int) 
    "smoothing by triangle filtering with reflecting boundaries"
    n = length(x)
    t = zeros(eltype(x),n+2*nb)
    for i in 1:n
        xi = x[i]/(nb*nb)
        t[i] -= xi
        t[i+nb] += 2*xi
        t[i+2*nb] -= xi
    end
    doubint!(t)
    return fold(t, nb)
end

# ╔═╡ 4f24b4a8-4538-4ec8-9be4-fe4ff44a1fab
function smooth(x::Array, nb::Vector{Int}) 
    "multidimensional smoothing"
    y = deepcopy(x)
    # loop over dimensions
    for dim in 1:length(nb)
        y = mapslices(slice -> smooth(slice, nb[dim]), y; dims=dim)
    end
    return y
end

# ╔═╡ 32991c88-a126-4efb-8e86-fb396f544e62
function smooth_division(numerator::Array, denominator::Array, 
	                     radius::Vector{Int}; niter=100, verb=true)
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
		            x -> smooth(x, radius), 1.0, numerator * norm, p0, niter; verb=verb)
end 

# ╔═╡ 95031504-241e-47cb-9ff5-507c194cac37
function find_radius(data1, data2, radius0, rad; niter=10)
	radius = deepcopy(radius0)
	for iter in 1:niter # Gauss-Newton iterations
		res = data2 - smoothr(data1, radius)
		@show iter, res ⋅ res
		der = smooth_der(data1, radius)
		radius = radius + smooth_division(res, der, [rad]; verb=false)
		radius[radius .< 1] .= 1.0
	end
	return radius
end

# ╔═╡ 6730a07d-f464-4a12-b618-15f61d303e74
der = smooth_der(data1, radius0)

# ╔═╡ 74f85d6c-2fb8-400e-a68f-1907e4f408b5
eradius = find_radius(data1, data2, radius0, 5; niter=40);

# ╔═╡ b8360c4a-9c9e-4fdf-8fe1-6618573bf40d
plot([radius eradius], label=["true" "estimated"], 
	 title="Smoothing Radius Estimation")

# ╔═╡ 2b96cbe8-63c3-4623-9eb3-369bdd71c43e
plot([data2 smoothr(data1, eradius)], label=["true" "estimated"], 
     title="Data Match")

# ╔═╡ 5fdbb3ba-9fe0-4a99-9643-a52671203063
md"""
## Sparseness regularization using the seislet transform

Promoting sparsity is an effective regularization method. To do so adaptively, we need a sparsifying transform for the presumed model pattern.

For a nonstationary signal with smoothly varying frequencies, the seislet transform is an appropriate tool.
"""

# ╔═╡ 0510bbe0-39df-41dd-add8-bc91cb3e46ca
function predict_forward(z::Vector{T}, t::T, i, j) where T <: Complex
	t2 = t
	for i2 in i:i+j-1
		t2 *= z[i2]
	end
	return t2
end

# ╔═╡ ec641563-28a3-4dd8-a3b7-78c9186de13e
function predict_backward(z::Vector{T}, t::T, i, j) where T <: Complex
	t2 = t
	for i2 in i+j-1:-1:i
		t2 /= z[i2]
	end
	return t2
end

# ╔═╡ bdfda5b8-8966-47cd-918c-548616cb9a6e
function seislet_haar!(x::Vector{T}, z::Vector{T}; 
	                   inv::Bool=false) where T <: Complex
    n = length(x) # Assume n is a power of 2
    if inv
        j = n ÷ 2
        while j >= 1
            for i in 1:2j:n-j
                x[i] -= predict_backward(z, x[i+j], i, j)/2
				x[i+j] += predict_forward(z, x[i], i,j)
            end
            j ÷= 2
        end
    else
        j = 1
        while j <= n ÷ 2
            for i in 1:2j:n-j
                x[i+j] -= predict_forward(z, x[i], i, j)
                x[i] += predict_backward(z, x[i+j], i, j)/2
            end
            j *= 2
        end
    end
end

# ╔═╡ 4687f2f5-3b58-4e9a-9ed6-b6301ba1bc46
function order_coefficients(c::Vector, n::Int) 
    "put wavelet coefficients in the right order"
    nc = length(c)
    x = Array{eltype(c)}(undef, n)
    ic = 1
    x[1] = c[1]
    j = nc ÷ 2
    while j >= 1                  
        for i in 1:2j:nc-j
            ic += 1
            x[ic] = c[i+j]
        end
        if ic >= n
            return x
        end
        j ÷= 2
    end
    return x
end

# ╔═╡ 88a86e90-c6a4-46b1-901a-2ed776c0fb12
function reorder_coefficients(x::Vector, nc::Int) 
    "inverse transformation to order_coefficients"
    n = length(x)
    c = Array{eltype(x)}(undef, nc)
    ic = 1
    c[1] = x[1]
    j = nc ÷ 2
    while j >= 1
        for i in 1:2j:nc-j
            if ic < n
                ic += 1
                c[i+j] = x[ic]
            else
                c[i+j] = zero(eltype(x))
            end
        end
        j ÷= 2
    end  
    return c
end

# ╔═╡ 3f692d82-bd08-4efd-b7c5-03fd37b42e22
function seislet(transform::Function, x::Vector{T}, z::Vector{T};
                 inv::Bool=false) where T <: Complex
    y = similar(x)
    n = length(x)
    # find the nearest power of two
    nt = 1
    while nt < n
        nt *= 2
    end
    if inv
        t = reorder_coefficients(x, nt)
    else
        t = vcat(x, zeros(T, nt-n)) # pad with zeros
    end
    transform(t, z; inv=inv)     
    if inv
        y = t[1:n] # truncate
    else  
        y = order_coefficients(t, n)
    end   
    return y
end

# ╔═╡ 324998a4-4259-4067-a8aa-207a39354657
begin
	t = collect(1:128)
	freq = 1 .- 0.004*t
	pfreq = cumsum(freq)
	z = exp.(im*freq)
	chirp = exp.(im*pfreq)
end

# ╔═╡ 98c367c9-7b8c-41db-9b57-e8b69c8c8cf4
plot(real(chirp), title="Chirp Signal", label=:none)

# ╔═╡ a38853a0-d598-4ecd-88bb-69e6708080b9
seis = seislet(seislet_haar!, chirp, z);

# ╔═╡ 0a1ac211-e186-4b19-ac9b-d6a951510af1
plot(real(seis), title="Seislet Transform", linewidth=2, 
	 color=:red, label=:none, xlabel="Scale")

# ╔═╡ 624f8465-1ec4-461f-b811-a915688b50d5
inv = seislet(seislet_haar!, seis, z; inv=true);

# ╔═╡ 8af493ef-74ef-4f72-b195-ba34b2ae3623
plot(real(inv), title="Inverse Seislet Transform", label=:none)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
FastMarching = "7c16e180-9f04-11e8-24a6-e7c7f74617b0"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
FastMarching = "~0.2.7"
HTTP = "~1.10.15"
Plots = "~1.40.11"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "5cbc598c3eac17cb1e06dc8aabda3b5fc899645b"

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
git-tree-sha1 = "2ac646d71d0d24b44f3f8c84da8c9f4d70fb67df"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.4+0"

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

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

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
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "e7b7e6f178525d17c720ab9c081e4ef04429f860"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.4"

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

[[deps.FastMarching]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "58174c1474002bd7ed48771b841b635b4ce97373"
uuid = "7c16e180-9f04-11e8-24a6-e7c7f74617b0"
version = "0.2.7"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

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
git-tree-sha1 = "c67b33b085f6e2faf8bf79a61962e7339a81129c"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.15"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "1d4015b1eb6dc3be7e6c400fbd8042fe825a6bac"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.10"

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
git-tree-sha1 = "d77592fa54ad343c5043b6f38a03f1a3c3959ffe"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.1+0"

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
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

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
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

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
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "24be21541580495368c35a6ccef1454e7b5015be"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.11"

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

[[deps.URIs]]
git-tree-sha1 = "cbbebadbcc76c5ca1cc4b4f3b0614b3e603b5000"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.2"

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

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

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
git-tree-sha1 = "82df486bfc568c29de4a207f7566d6716db6377c"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.43+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e6f1a4f245f66f93f28e55879f9ba47fed66f36"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.0+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

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
# ╟─e88510fb-4681-4bf7-bcf1-7cfad17a2559
# ╟─c86323f6-9cc4-434d-b147-f4107fa61a39
# ╟─11102582-1e4f-11f0-2558-65ff8aa2a1f8
# ╟─a238973b-e8f4-40c0-9d60-7194d4871438
# ╟─8e3b79dd-0af4-4710-b2e1-e3ac1aab1ac0
# ╟─96c62806-16a4-4b69-a02e-5a7d42ccd065
# ╠═e6b8f7b9-29b5-48be-b8c1-e0105986eb56
# ╠═44f87835-008d-4670-84d7-94940895a565
# ╠═64bd1fbc-9a20-4902-afda-b4f6d04c8094
# ╠═b69f07b7-6857-46f4-8dd6-0afc720c0a5e
# ╠═231316b2-3d2c-4898-a15b-9262cac03294
# ╠═e856d9de-5b21-4276-bd7d-8c8cab684827
# ╠═0abe931a-737f-408a-bdec-7a41bf014a2b
# ╟─41db06e2-a9aa-4d6d-a378-d01d8f5e2a13
# ╠═a5e515ab-5a1b-46a6-8288-f5ea3a5bfde8
# ╠═ce5a04e4-f8b0-487d-a3af-6f4f720e4570
# ╠═36dd5738-1419-48e4-859b-532d3638a658
# ╠═48a90528-5527-4c17-8a69-d86c07e1314f
# ╠═3f828447-5a33-42ab-a888-7c8480336e72
# ╠═7244a1da-8ba8-4546-8089-eb8561af3471
# ╠═547a6bc6-9c75-4c16-9c66-4c8844a1d91f
# ╠═e1d2dced-b59b-4bed-a64b-b005f03c9397
# ╠═e73dbb85-d21b-47bf-b199-1a15c515b95c
# ╠═4537b2d6-90c7-48c0-9a29-eb592c699fad
# ╠═b3e3c1e5-cb8c-4e4b-b49d-a25084de0081
# ╠═d219e674-320f-46d7-a3b5-b122c2be77a0
# ╠═a1f920ad-c494-424b-96ea-2d886d05bee9
# ╠═d025636b-8b39-4441-a7e8-c540c4a332db
# ╟─2fe180bb-ffa7-4f25-b162-7a946c3e83e2
# ╠═36308986-5642-436f-885c-5fb7b33ed0f4
# ╠═b75a4dd3-cf6a-4bf7-869f-60de08ad492c
# ╠═d78fffcd-9d17-424d-887b-fa3cc448e570
# ╠═d2bd20d0-d665-4ae1-b9ab-99596317b9af
# ╠═6cd9e01c-4542-4ea7-a128-6856a0a4a163
# ╠═0ce44c53-7be7-4c58-8743-c8ec8424aa74
# ╠═b649d55a-4def-4c98-bd21-19be81dc3f62
# ╟─62682722-5fea-4449-a7d4-1f8973515ee4
# ╟─359c5695-45bb-4588-9f60-e5df5d394bb5
# ╟─50f6fcaa-10e4-41bc-aec3-5fd86b5f0775
# ╟─516fe177-9a8a-406b-99c7-245449b7066c
# ╟─63f60cf5-8e11-41be-ba30-fbf357b8eb87
# ╠═ebf4263e-5f8c-42f0-bedf-7d5e163a1d85
# ╟─020ea553-a7a5-457c-9db5-19859fca4592
# ╠═17ee38ba-49bf-4083-9b67-2f04f2d0e8f3
# ╠═a3100bd3-f0ea-4db1-a128-5ee8ee99b4cc
# ╟─6ad926d8-d80a-46b4-a483-ebc5c5caa8c6
# ╟─2e3e178a-bc18-46e2-b118-4634987a8a1a
# ╟─4f24b4a8-4538-4ec8-9be4-fe4ff44a1fab
# ╟─32991c88-a126-4efb-8e86-fb396f544e62
# ╠═95031504-241e-47cb-9ff5-507c194cac37
# ╠═6730a07d-f464-4a12-b618-15f61d303e74
# ╠═74f85d6c-2fb8-400e-a68f-1907e4f408b5
# ╠═b8360c4a-9c9e-4fdf-8fe1-6618573bf40d
# ╠═2b96cbe8-63c3-4623-9eb3-369bdd71c43e
# ╟─5fdbb3ba-9fe0-4a99-9643-a52671203063
# ╠═0510bbe0-39df-41dd-add8-bc91cb3e46ca
# ╠═ec641563-28a3-4dd8-a3b7-78c9186de13e
# ╠═bdfda5b8-8966-47cd-918c-548616cb9a6e
# ╟─4687f2f5-3b58-4e9a-9ed6-b6301ba1bc46
# ╟─88a86e90-c6a4-46b1-901a-2ed776c0fb12
# ╠═3f692d82-bd08-4efd-b7c5-03fd37b42e22
# ╠═324998a4-4259-4067-a8aa-207a39354657
# ╠═98c367c9-7b8c-41db-9b57-e8b69c8c8cf4
# ╠═a38853a0-d598-4ecd-88bb-69e6708080b9
# ╠═0a1ac211-e186-4b19-ac9b-d6a951510af1
# ╠═624f8465-1ec4-461f-b811-a915688b50d5
# ╠═8af493ef-74ef-4f72-b195-ba34b2ae3623
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
