### A Pluto.jl notebook ###
# v0.20.27

#> [frontmatter]
#> chapter = 4
#> section = 11
#> order = 11
#> image = "https://raw.githubusercontent.com/fonsp/Pluto.jl/580ab811f13d565cc81ebfa70ed36c84b125f55d/demo/plutodemo.gif"
#> title = "11 Data Analysis Notebook"
#> tags = ["module4", "track_julia", "track_material", "Pluto", "PlutoUI"]
#> layout = "layout.jlhtml"

using Markdown
using InteractiveUtils

# ╔═╡ 00b5dbb6-8d51-4522-8ddc-0ce68d48119b
begin
	using PlutoUIExtra
	TableOfContents()
end

# ╔═╡ e2539189-203d-4482-900f-21fddfba4562
using HTTP

# ╔═╡ bf8bd7d5-63b3-43af-99d8-2f79e9938a0c
using Plots, LaTeXStrings

# ╔═╡ 9a54ae44-a22e-4d19-90bd-63985cdc0689
using FFTW

# ╔═╡ 518ad9cf-996d-4a16-984e-5d30404d5033
using BandedMatrices

# ╔═╡ 98270df2-690f-4452-81e9-021fa678abb2
using BenchmarkTools

# ╔═╡ c0ede6d0-2b24-4519-b69a-30cb799b4a16
let
    notebook_dir = @__DIR__
    toc_path = joinpath(notebook_dir, "TOC_Notebook.jl")
    
    Markdown.parse("[⬅️ Back to Table of Contents](./open?path=$(HTTP.escapeuri(toc_path)))")
end

# ╔═╡ 4dcd539e-2e93-42ae-8b38-654d3a0e15f1
md"""
# Forward interpolation

Data gridding, the process of interpolating irregularly sampled data onto a regular grid, is a fundamental challenge in geophysical data analysis. Before tackling this problem, we will first consider a simpler one: how to interpolate data from samples on a regular grid. This problem lies at the intersection of numerical analysis and digital signal processing.
"""

# ╔═╡ fa60c4a1-81f3-4db3-b289-08ed7b52b57c
md"""
## Forward interpolation as a linear operator
"""

# ╔═╡ 7f8a931d-367f-440c-ad4b-7bac90377fd7
function plot_sketch(coord)
    plt = plot([0, 4],[0, 0], label=:none, color=:blue)
    for k in 1:5
        plot!(plt, [k-1, k-1], [-0.1, 0.1], label=:none, color=:blue)
        annotate!(plt, k-1, -0.2, L"x_%$k")
    end
    for k in 1:length(coord)
        plot!(plt, [coord[k], coord[k]], [0.2, 0], arrow=:closed, linewidth=2,
              label=:none, color=:red, aspect_ratio=:equal, ylim=(-0.2, 0.4), border=:none)
        annotate!(plt, coord[k], 0.3, L"y_%$k")
    end
    return plt
end

# ╔═╡ f8984287-0562-4983-b50d-9ed63a540956
begin
	coord=[0.5, 2.0, 2.7, 3.1]
	plot_sketch(coord)
end

# ╔═╡ 10f0c8f2-64b5-4213-83c1-4f1618d404b6
md"""
Consider the problem of interpolating a 1-D vector $\mathbf{x}$ measured on a regularly sampled grid into an irregularly sampled vector $\mathbf{y}$. If we choose the nearest point on the grid, the transformation between $\mathbf{x}$ and $\mathbf{y}$ can be represented as a matrix multiplication

$$\left[\begin{array}{l} y_1 \\ y_2 \\ y_3 \\ y_4\end{array}\right] =
\left[\begin{array}{ccccc} 
1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 & 0
\end{array}\right]\,
\left[\begin{array}{l} x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5\end{array}\right]\;.$$
"""

# ╔═╡ 56317420-6502-4cd9-9bb0-3072c63cf304
md"""
If, instead of simply taking the nearest point, we linearly interpolate between two neighboring points, the matrix will take the form

$$\left[\begin{array}{l} y_1 \\ y_2 \\ y_3 \\ y_4\end{array}\right] =
\left[\begin{array}{ccccc} 
0.5 & 0.5 & 0   & 0   & 0 \\
0   & 0   & 1.0 & 0   & 0 \\
0   & 0   & 0.3 & 0.7 & 0 \\
0   & 0   & 0   & 0.9 & 0.1
\end{array}\right]\,
\left[\begin{array}{l} x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5\end{array}\right]\;.$$
"""

# ╔═╡ 52a70978-1070-4567-aa15-5a0e3b925ce4
function nnint(regul::Vector{T}, coord, d1=1, o1=0) where T <: Real
    "Nearest-neighbor interpolation"
    nm, nd = length(regul), length(coord)
    irreg = similar(coord)
    for id in 1:nd
        f = 1 + (coord[id] - o1)/d1     
        im = round(Int, f) # nearest integer less or equal to f   
        irreg[id] =  (0 < im && im <= nm) ? regul[im] : zero(T)
    end
    return irreg
end

# ╔═╡ 46be2055-cfe3-4911-8d26-4202bf4d0187
function lint(regul::Vector{T}, coord, d1=1, o1=0) where T <: Real
    "Linear interpolation"
    nm, nd = length(regul), length(coord)
    irreg = similar(coord)
    for id in 1:nd
        f = 1 + (coord[id] - o1)/d1     
        im = floor(Int, f) # nearest integer less or equal to f
        f -= im  
        irreg[id] =  (0 < im && im < nm) ? 
            (1-f) * regul[im]  +  f * regul[im+1] : zero(T)
    end
    return irreg
end

# ╔═╡ 1c6dad36-cb4d-4cea-a750-5bdcd541a283
x = rand(5)

# ╔═╡ aedcea05-f476-447e-aa98-41bec07088ca
nnint(x, coord)

# ╔═╡ dd2b203f-3ae7-4d3b-bea0-21823fc97090
lint(x, coord)

# ╔═╡ d6cd774a-8902-42bf-8320-2487a2fe47f3
md"""
Is there a systematic way to improve the accuracy of interpolation?
"""

# ╔═╡ 6a7adaf3-203d-43b9-9607-04acb9411cfe
md"""
## Interpolation and sampling

A function $y(x)$ defined on the interval $[-\pi, \pi]$ can be extended periodically and represented by an infinite Fourier series

$$y(x) = \displaystyle \sum\limits_{n=-\infty}^{\infty} c_n\,e^{-i\,n\,x}\;,$$

where the complex coefficients $c_n$ are defined as

$$c_n = \displaystyle \frac{1}{2\pi}\,
\int\limits_{-\pi}^{\pi} y(x)\,e^{i\,n\,x}\,dx\;.$$

"""

# ╔═╡ 28d7148f-0ad8-44e6-9e32-84504429aa5f
md"""
Suppose we substitute the Fourier transform of a band-limited function $F(\omega)$ for $y(x)$. In that case, $c_n$ will correspond to the sampled values of the time-domain function $f(t)$. We can rewrite the series as

$$F(\omega) = \displaystyle \sum\limits_{n=-\infty}^{\infty}
f(n)\,e^{-i\,n\,\omega}\;.$$
"""

# ╔═╡ c4d0b77d-683d-478d-bdab-a7f7460d4f37
md"""
Taking it to the time domain with the inverse Fourier transform,

$$\begin{array}{rcl}f(t) & = & \displaystyle \sum\limits_{n=-\infty}^{\infty}
\frac{f(n)}{2\pi}\,\int\limits_{-\pi}^{\pi} e^{i\,(t-n)\,\omega}\,d\omega \\
& = & \displaystyle \sum\limits_{n=-\infty}^{\infty} f(n)\,
\frac{\sin[\pi\,(t-n)]}{\pi\,(t-n)} \\ & = & \displaystyle  \sum\limits_{n=-\infty}^{\infty} f(n)\,\mbox{sinc}[\pi\,(t-n)]\;.\end{array}$$
"""

# ╔═╡ 2ec3dd22-b968-4c8d-adc1-d972e8fa304c
md"""
The last equation is the famous *sampling theorem*. It shows that a band-limited function can be fully reconstructed from its samples on a regular grid using the sinc function as an interpolation kernel.


* **Shannon, C. E., 1949, Communication in the presense of noise: Proc. I.R.E., 37, 10–21.**
"""

# ╔═╡ c4cc29f7-d49a-4920-84cb-c23dedaf2601
sinc(x)=sin(π*x)/(π*x)

# ╔═╡ 38c99849-5a4e-44a8-acd9-5f3513b6bc95
heaviside(x) = x > 0 ? 1.0 : 0.0

# ╔═╡ 8da05bb9-ae1a-44d4-a818-1ff29eea8560
box(ω)=heaviside(ω + π) - heaviside(ω - π)

# ╔═╡ b0607485-0d7b-48f1-86cd-f1b8ea93587f
function plot_tf(time, freq, name)
    ptime=plot(time, -2π, 2π, linewidth=3, ylim=(-0.3,1.1), 
               label=:none, xlabel="time");
    pfreq=plot(freq, -2π, 2π, linewidth=3, ylim=(-0.3,1.1), 
               label=:none, xlabel="frequency",color=:red)
    return plot(ptime, pfreq, layout=(1, 2), plot_grid_title=name)
end

# ╔═╡ 366ffd53-1d78-4cd0-aba0-83033fadaffb
plot_tf(sinc, box, "Sinc Interpolator")

# ╔═╡ 5ba37030-9389-40c1-9004-5c4220cd8dd0
md"""
The ideal sinc interpolator is a valid theoretical concept, but it is impractical because of its slow decay, as discussed earlier. One practical approach is to taper the sinc function to make it more compact. However, more accurate and systematic methods exist for constructing interpolation weights.
"""

# ╔═╡ 7814e606-cc43-458e-90bf-51e5fd96b09f
md"""
## Forward interpolation as filtering

Let's examine the interpolation matrix more closely. Each row of the matrix can be interpreted as the result of a correlation with a short filter that shifts the input trace by a non-integer multiple of the sampling interval. Interpolation involves a non-stationary correlation that can be implemented using filters that shift traces. The ideal shifting filter is $Z^{s}=Z^n Z^{\sigma}$, where $s=n+\sigma$ is the shift in samples, $n$ is the integer part of $s$, and $\sigma$ is the fractional remainder. Practical interpolators use various approximations to $Z^{\sigma}$.
"""

# ╔═╡ 72042f22-222a-4f0e-a563-ab7a425c1eff
md"""
For example, nearest-neighbor interpolation corresponds to the trivial approximation
$$Z^{\sigma} \approx 1\;.$$

In the continuous world, this corresponds to approximating a function using a superposition of box functions

$$f(t) \approx \sum\limits_n f(n)\,\beta_0(t-n)\;,$$

where
$$\beta_0(t) = \left\{\begin{array}{rcl} 
1 & \quad for & |t| \le 1/2 \\
0 & \quad for & |t| > 1/2 \end{array}\right.$$
"""

# ╔═╡ 61e6c5ff-2336-4cba-b463-c7ae6cfc3d53
tbox(x)=heaviside(x+1/2)-heaviside(x-1/2)

# ╔═╡ 60fa6edc-0945-4dd5-8a77-b3eb5fb94180
nnf(ω)=sin(ω/2)/(ω/2)

# ╔═╡ 2534e7e6-2f81-4a7b-ad67-10378e34d8b2
plot_tf(tbox, nnf, "Nearest Neighbor Interpolator")

# ╔═╡ e8da4381-5433-4b8d-a8af-fdf8a709c885
md"""
The linear interpolation corresponds to the linear approximation at zero frequency (or, equivalently, $Z=1$):

$$Z^{\sigma} = \left[1+(Z-1)\right]^\sigma \approx 1+\sigma\,(Z-1) = (1-\sigma)+\sigma\,Z\;.$$

In the continuous world, this corresponds to approximating a function using a superposition of triangle functions:

$$f(t) \approx \sum\limits_n f(n)\,\beta_1(t-n)\;,$$
where
$$\beta_1(t) = \left\{\begin{array}{rcl} 
1-|t| & \quad for & |t| \le 1 \\
0 & \quad for & |t| > 1 \end{array}\right.$$
"""

# ╔═╡ 35567009-0735-4ec1-bdac-db34f48d3544
hat(x)=(heaviside(x+1)-heaviside(x-1))*(1-abs(x))

# ╔═╡ 5dd7e0ee-321b-4e7c-98d5-aaf2274ad301
linf(ω) = (2/ω*sin(ω/2))^2

# ╔═╡ d3e60bd7-ffd9-44d4-b565-59adf21127cb
plot_tf(hat, linf, "Linear Interpolator")

# ╔═╡ 1cf5fdb9-d174-4278-9d32-36af39c21742
tmax = 80

# ╔═╡ 790eea28-2383-4c7c-afd8-c3ecbf540b15
function chirp(x::Vector{T}) where T <: Real
    xx = reshape(x, 1, :)
    yy = reshape(x, :, 1)
    f = (0.25 * xx.^2 .+ yy.^2) ./ (0.25 * tmax)^2
    return 0.5 * cos.(8f) .* exp.(-f)
end

# ╔═╡ ca57cb3d-7c99-438f-bea2-646bf815955a
begin
	xc(n) = [((i-1)/(n-1) - 1/2)*tmax for i in 1:n]
	xdense = xc(500)
	xsparse = xc(50)
	dense = chirp(xdense)
	sparse = chirp(xsparse)
end

# ╔═╡ 86747eee-a356-45d8-820b-77d29efe2d1b
begin
	dx=xsparse[2]-xsparse[1]
	x0=xsparse[1]
end

# ╔═╡ 126a54af-bfa0-47a7-a9e5-cd8ecbcfdd80
begin
	p1 = heatmap(xdense,xdense,dense, color=:grays, title="Ideal")
	p2 = heatmap(xsparse,xsparse,sparse, color=:grays, title="Decimated")
	plot(p1, p2, layout=(1, 2))
end

# ╔═╡ c8c361d0-30fe-483e-92d7-2eb4e697dbc9
begin
	interp = Dict("Nearest Neighbor" => nnint, "Linear" => lint)
	plots1 = Dict{String, Plots.Plot}()
	plots2 = Dict{String, Plots.Plot}()
	for (name, func) in interp
	    in1 = mapslices(x -> func(x, xdense, dx, x0), sparse; dims=1)
	    in2 = mapslices(x -> func(x, xdense, dx, x0), in1; dims=2)
	    
	    error = in2 - dense
	    
	    plots1[name] = plot(xdense, error[:,250], ylim=[-0.15,0.15],
	                        title="$name Interpolation Error", 
	                        label=:none, color=Int(name[1]))
	    plots2[name] = heatmap(xdense, xdense, error, color=:grays, 
	                           title=name)
	end
end

# ╔═╡ 55a63a5e-15e8-48ff-bc86-105bc74d631d
plot(plots1["Nearest Neighbor"], plots1["Linear"], layout=(2, 1))

# ╔═╡ 12c5556a-2b22-4a81-8a73-d9cd14d5401c
plot(plots2["Nearest Neighbor"], plots2["Linear"], layout=(1, 2))

# ╔═╡ 18bc745a-83f6-41e8-9f42-735079e1d7a0
md"""
Nearest-neighbor and linear interpolation perform well on low-frequency or oversampled data. More accurate methods are required to preserve high-frequency details, such as those found in typical seismic data.

There are two different ways to extend the approximation sequence for longer, more accurate filters.
"""

# ╔═╡ 6c848cd8-06c8-44b1-8a42-465405f485b9
md"""
## Lagrange filters

If we expand the term $\left[1+(Z-1)\right]^\sigma$ into a longer Taylor series around $Z=1$, we can obtain progressively longer FIR (finite impulse response) filters whose coefficients are polynomial in $\sigma$. For example, the next-order expansion is

$$\begin{array}{rcl}Z^{\sigma} & = & \left[1+(Z-1)\right]^\sigma \\ & \approx & \displaystyle 1+\sigma\,(Z-1)+\frac{\sigma\,(\sigma-1)}{2}\,(Z-1)^2\end{array}$$
"""

# ╔═╡ c8135466-c718-456d-995e-fc8f6efe008c
md"""
or making it more symmetric,

$$\begin{array}{rcl}
\nonumber
Z^{\sigma} & = & Z^{-1}\,\left[1+(Z-1)\right]^{\sigma+1} \\
\nonumber
& \approx & \displaystyle
Z^{-1}\,\left[1+(\sigma+1)\,(Z-1)+
\frac{\sigma\,(\sigma+1)}{2}\,(Z-1)^2\right] \\
& = & \displaystyle
\frac{\sigma\,(\sigma-1)}{2}\,Z^{-1} + 
(1-\sigma^2)+\frac{\sigma\,(\sigma+1)}{2}\,Z\;.\end{array}$$

Note that the coefficients of the three-point filter are quadratic polynomials in $\sigma$.
"""

# ╔═╡ 3977adb5-fcda-4df2-b2b1-aa9c029c4be0
md"""
Extending this further, we can derive FIR (finite impulse response) filters of the form

$$Z^{\sigma} \approx \displaystyle \sum\limits_{n=-N/2+1}^{N/2} L_n(\sigma)\,Z^n\;,$$

where $L_n(\sigma)$ is the *Lagrange* polynomial

$$L_n(\sigma) = \displaystyle \prod\limits_{k \ne n} \frac{(\sigma-k)}{(n-k)}\;.$$
"""

# ╔═╡ 1df4e9a9-1d71-4b3d-938b-35c220e8cf45
function int1(nw::Int, interpolator::Function, 
              regul::Vector{T}, coord, d1=1, o1=0) where T <: Real
    "1-D generic interpolation"
    nm, nd = length(regul), length(coord)
    irreg = zeros(T,nd)
    @inbounds for id in 1:nd
        f = 1 + (coord[id] - o1)/d1     
        im = floor(Int, f) # nearest integer less or equal to f
        w = interpolator(f-im, nw)
        for iw in 1:nw
            ir = im - nw÷2 + iw
            ir = max(1, min(ir, nm))
            irreg[id] += w[iw] * regul[ir]
        end
    end
    return irreg
end

# ╔═╡ 8d014cdb-80e0-4f5a-916b-1a51401aeeea
linear(σ, nw) = [1-σ, σ]

# ╔═╡ b2d94e09-4c47-4351-be46-1cdaf9b42533
@assert lint(x, coord) ≈ int1(2, linear, x, coord)

# ╔═╡ f6d86659-a232-412d-8eb8-067b306b90d5
function lagrange(σ, nw) 
    "Lagrange interpolator"
    nc = (nw + 1) ÷ 2
    w = ones(nw)
    for i in 1:nw
        ξ = σ + nc - i
        for j in 1:nw
            if i != j
                w[i] *= (1 + ξ / (i - j))
            end
        end
    end
    return w
end

# ╔═╡ c8e6830c-f05e-44d7-91ea-4e68330bd8f5
@assert lint(x, coord) ≈ int1(2, lagrange, x, coord)

# ╔═╡ b89ab958-279a-4614-9b9a-d15593c293db
md"""
!!! assignment
    ## Task 1

    Let’s compare various interpolators by plotting them as filters together with their spectra.
"""

# ╔═╡ 2aea9ad8-b3e8-4f36-88df-801277b7dd13
function plot_interpolator(interpolator::Function, nw::Int)
    σ = 0.7 # example shift
    w = interpolator(σ, nw)
    name = uppercasefirst(String(Symbol(interpolator)))
    T = eltype(w)
    # pad with zeros
    wpad = vcat(zeros(T,32-nw÷2), w, zeros(T,32+nw÷2-nw))
    # plot with stems
    p1=plot(zeros(T, 64), label=:none, color=:black)
    plot!(p1, wpad, line=:stem, marker=:circle, border=:none,
          title="$name-$nw Filter for σ=$σ", label=:none)
    # plot spectrum
    f = rfftfreq(64, 1)
    spectrum = abs.(rfft(wpad))
    p2 = plot(f, spectrum, label=:none, title="Spectrum", xlabel="frequency")
    # display side by side
    return plot(p1, p2, layout=(1, 2))
end

# ╔═╡ 33c533dd-ac97-4b71-b0c1-6734b53ce9c4
plot_interpolator(linear, 2)

# ╔═╡ 3e708a91-dee9-4930-9ce9-b47378271f08
plot_interpolator(lagrange, 2)

# ╔═╡ a0708eb3-0fb0-4f6f-8446-6d0509645025
md"""
Plot the Lagrange filters of different sizes and compare the results.
"""

# ╔═╡ df0cb1aa-12cb-4760-a6f3-3f4716df4e73
md"""
## Cubic convolution

The cubic convolution filter can be expressed as a polynomial

$$\begin{array}{rcl}
\nonumber
Z^{\sigma} & \approx & \displaystyle -\frac{\sigma\,(1-\sigma)^2}{2}\,Z^{-1} \\
& & + \displaystyle \frac{(1-\sigma)\,(2 + 2\,\sigma - 3 \sigma^2)}{2} \\
&  & + \displaystyle \frac{\sigma\,(1 + 4\,\sigma - 3\,\sigma^2)}{2}\,Z \\
& & - \displaystyle \frac{(1-\sigma)\,\sigma^2}{2}\,Z^2\end{array}$$
"""

# ╔═╡ 9775af25-c525-4656-a5b6-6f8af95afcfe
md"""
and is designed to approximate the ideal sinc-function interpolator with a four-point FIR filter.

* **Keys, R. G., 1981, Cubic convolution interpolation for digital image processing: IEEE Trans. Acoust., Speech, Signal Process., ASSP-29, 1153–1160.**
"""

# ╔═╡ a1f6c924-0513-4ed4-8cf3-724de0e316c6
function cubic_convolution(σ, nw)
    @assert nw==4
    return [-σ*(1 - σ)^2/2,
        (1 - σ)*(2 + 2σ - 3σ^2)/2,
        σ*(1 + 4σ - 3σ^2)/2,
        (σ-1)*σ^2/2]
end

# ╔═╡ 64f89d67-bee5-431d-9c69-2fb008c054e4
begin
	interp4 = Dict("Lagrange-4" => lagrange, 
	              "Cubic Convolution" => cubic_convolution)
	plots41 = Dict{String, Plots.Plot}()
	plots42 = Dict{String, Plots.Plot}()
	for (name, func) in interp4
	    in1 = mapslices(x -> int1(4, func, x, xdense, dx, x0), 
	          sparse; dims=1)
	    in2 = mapslices(x -> int1(4, func, x, xdense, dx, x0), 
	          in1; dims=2)
	    
	    error = in2 - dense
	    
	    plots41[name] = plot(xdense, error[:,250], ylim=[-0.15,0.15], 
	                        title="$name Interpolation Error", 
	                        label=:none, color=Int(name[1]))
	    plots42[name] = heatmap(xdense, xdense, error, 
	                           color=:grays, title=name)
	end
end

# ╔═╡ cdd48d73-174e-4ec9-aa58-8d8795594470
plot(plots41["Lagrange-4"], plots41["Cubic Convolution"], layout=(2, 1))

# ╔═╡ e7f12c0b-c906-45aa-b85a-ea10f84c0157
plot(plots42["Lagrange-4"], plots42["Cubic Convolution"], layout=(1, 2))

# ╔═╡ 55943315-aa6f-4f57-a071-12735691ca54
md"""
!!! assignment
    ## Task 2

    Using the example above, calculate the maximum absolute errors for Lagrange-4 and cubic convolution and compare them. Which of the two methods is more accurate?
"""

# ╔═╡ 279a9b3d-3770-4139-8a1a-9a5b3a25f2d6
md"""
## B-spline filters

Noticing that the triangle function $\beta_1(x)$ is the continuous convolution of the box function $\beta_0(x)$, we might ask what would happen if we kept convolving the box function with itself. This process produces piecewise-polynomial *B-spline* functions, which can be explicitly defined as
"""

# ╔═╡ cf3761ff-2063-4615-88f2-e2acc72177ed
md"""
$$\beta_n(x) = 
\frac{1}{n!}\,\sum_{k=0}^{n+1} C_k^{n+1} (-1)^k 
(x + \frac{n+1}{2} - k)_{+}^n\;,$$

where $C_k^{n+1}$ are the binomial coefficients,
$$C_k^n = \displaystyle \frac{n!}{k!\,(n-k)!}\;,$$

and the function $x_{+}$ is defined as follows:
$$x_{+} = \left\{\begin{array}{lcr}
x, & \mbox{for} & x > 0 \\
0, & \mbox{otherwise} &
\end{array}\right.$$

This equation can be proven by induction.
"""

# ╔═╡ a4471955-bf4c-4ea1-914f-dc9f97d92eba
md"""
A prevalent choice in practice is the cubic B-spline, which has the expression corresponding to $n=3$:

$$\beta_3(x) = \left\{\begin{array}{ll} \displaystyle 
\left(4-6|x|^2+3 |x|^3\right)/6, & \mbox{for} \quad  1 > |x| \geq 0
  \\ \displaystyle (2-|x|)^3/6, & \mbox{for} \quad 2 > |x| \geq 1 \\ 0, &
  \mbox{elsewhere} 
\end{array}\right.$$
"""

# ╔═╡ c787dc4a-7fa5-4abe-b0ab-8c561bc121b0
md"""
The Fourier transform of a B-spline of order $n$ is the $(n+1)$-th power of the sinc function:

$$B_n(\omega)=\displaystyle\left[\frac{2}{\omega}\,\sin\left(\frac{\omega}{2}\right)\right]^{n+1}.$$

As previously discussed, this function rapidly approaches a Gaussian as $n$ increases. Therefore, $\beta_n(x)$ also approximates a Gaussian.
"""

# ╔═╡ a07489a0-0120-4e35-b3b5-aef42e85c1e9
begin
	z1(x)=(4 - 6*x^2 + 3*x^3)/6
	z2(x)=(2 - x)^3/6
	spl3(x)=(heaviside(x + 1) - heaviside(x - 1))*z1(abs(x)) +
	        (heaviside(x - 1) - heaviside(x - 2))*z2(x) +
	        (heaviside(x + 2) - heaviside(x +1 ))*z2(-x)
end

# ╔═╡ 77c19c98-8d79-47bb-81ae-1508b571ff20
begin
	splf3(ω)=(2/ω*sin(ω/2))^4
	isplf3(ω)=splf3(ω)/(z1(0) + 2*z1(1)*cos(ω))
end

# ╔═╡ b65f6736-8b33-4060-af1f-12c63506fa97
plot_tf(spl3, splf3, "B-Spline Basis Function Order 3")

# ╔═╡ 13c660d1-8847-4dfd-803e-a8c51f5dd6e0
begin
	zz1(x)=(2416 - 1680*x^2 + 560*x^4 - 140*x^6 + 35*x^7)/5040
	zz2(x)=(2472 - 392*x - 504*x^2 - 1960*x^3 + 2520*x^4 - 1176*x^5 + 252*x^6 - 21*x^7)/5040
	zz3(x)=(-1112 + 12152*x - 19320*x^2 + 13720*x^3 - 5320*x^4 + 1176*x^5 - 140*x^6 + 7*x^7)/5040
	zz4(x)=(4 - x)^7/5040
	spl7(x)=(heaviside(x + 1) - heaviside(x - 1))*zz1(abs(x))+
	        (heaviside(x - 1) - heaviside(x - 2))*zz2(x)+
	        (heaviside(x - 2) - heaviside(x - 3))*zz3(x)+
	        (heaviside(x - 3) - heaviside(x - 4))*zz4(x)+
	        (heaviside(x + 2) - heaviside(x + 1))*zz2(-x)+
	        (heaviside(x + 3) - heaviside(x + 2))*zz3(-x)+
	        (heaviside(x + 4) - heaviside(x + 3))*zz4(-x)
end

# ╔═╡ c1226bf1-b577-457b-9f59-7c3a0942096a
begin
	splf7(ω)=(2/ω*sin(ω/2))^8
	isplf7(ω)=splf7(ω)/(zz1(0) + 2*zz2(1)*cos(ω) + 2*zz3(2)*cos(2*ω) + 2*zz4(3)*cos(3*ω))
end

# ╔═╡ d41688fa-870e-4d84-80a3-cc6e46faeb78
plot_tf(spl7, splf7, "B-Spline Basis Function Order 7")

# ╔═╡ 49a774b4-3c6f-4bb3-b613-6a2224428445
md"""
If we represent a function as a superposition of B-splines

$$f(t) \approx \sum\limits_n c_n\,\beta_k(t-n)\;,$$

then, at sampled points,

$$f(n) \approx \sum\limits_m c_m\,\beta_k(n-m)\;,$$

which is simply a digital convolution of the signal $f(n)$ with sampled values of $\beta_k$. In the Z-transform notation,

$$F(Z) = C(Z)\,B_k(Z)\;,$$
"""

# ╔═╡ 9da99ac0-6b82-41fc-a0bd-91aa83df1801
md"""
which suggests finding the spline coefficients $c_m$ by inverse
filtering (polynomial division) 

$$C(Z)=\displaystyle \frac{F(Z)}{B_k(Z)}\;.$$ 

Once the coefficients are determined, a function can be interpolated at any location. For example, cubic spline interpolation requires inverse filtering with

$$B_3(Z) = Z^{-1}/6 + 2/3 + Z/6\;,$$

which can be factored as the autocorrelation of a two-point minimum-phase filter.
"""

# ╔═╡ 3a1493cb-bc93-48cb-a782-372549256060
md"""
Thus, B-spline interpolation of order $k > 1$ involves IIR (infinite impulse response) filtering and approximates the ideal interpolator as

$$Z^{\sigma} \displaystyle \approx {\frac{\displaystyle \sum\limits_{n=-N/2+1}^{N/2} C_n(\sigma)\,Z^n}{B(Z)}}\;.$$
"""

# ╔═╡ bbdea01f-6911-46eb-af3b-cfe779832b21
function conv(x::Vector{T}, b::Vector{T}) where T <: Real
    # convolution
    nx, nb = length(x), length(b)
    y = zeros(T, nx + nb - 1)
    for ib in 1:nb, ix in 1:nx
        y[ix + ib - 1] += x[ix] * b[ib] 
    end
    return y 
end

# ╔═╡ d8a7131d-0bf1-462d-8cfd-d3cc5bc5f8f4
begin
	a3 = 2 - sqrt(3)
	c3 = (2+ sqrt(3))/6
	N = 20
	F = [(-1)^k*a3^k for k in 0:N]
	FF3 = conv(F,reverse(F))/c3
	ispl3(x)=sum([FF3[N + 1 + k]*spl3(x-k) for k in -10:10])
end

# ╔═╡ 11b69044-f0f6-4267-9047-a01bc2805c58
plot_tf(ispl3, isplf3, "B-Spline Interpolator Order 3")

# ╔═╡ 7deafb32-da00-4744-ac50-d3813cacf9e4
begin
	a7=0.535281
	b7=0.122555
	c7=0.00914759
	d7=0.330597
	Fa = [(-1)^k*a7^k for k in 0:N]
	Fb = [(-1)^k*b7^k for k in 0:N]
	Fc = [(-1)^k*c7^k for k in 0:N]
	FF7 = conv(conv(conv(conv(conv(Fa, Fb), Fc), 
	     reverse(Fc)), reverse(Fb)), reverse(Fa))/d7
	ispl7(x)=sum([FF7[3*N + 1 + k]*spl7(x - k) for k in -10:10])
end

# ╔═╡ 1b9527d7-a303-4f2c-ad6b-5643e52f4936
plot_tf(ispl7, isplf7, "B-Spline Interpolator Order 7")

# ╔═╡ e1e12986-18b6-4a45-b738-b67fca8f706a
function spline(σ, nw)
    "B-spline interpolator"
    if nw == 4
        return [ 
        (1 + σ*((3 - σ)*σ-3))/6,
        (4 + 3*(σ -2)*σ^2)/6,
        (1 + 3*σ*(1 + (1 - σ)*σ))/6,
        σ^3/6 ]
    elseif nw == 8
        return [
        (1 + σ*(σ*(21 + σ*(σ*(35 + σ*((7 - σ)*σ-21))-35))-7))/5040,
        (120 + 7*σ*(σ*(72 + σ*(σ^2*(12 + σ*(σ-6))-40))-56))/5040,
        (1191 + 7*σ*(σ*(45 + σ*(95 + 3*σ*(σ*((5 - σ)*σ-5)-15)))-245))/5040,
        (2416 + 35*σ^2*(σ^2*(16 + σ^2*(σ-4))-48))/5040,
        (1191 + 35*σ*(49 + σ*(9 + σ*(σ*(σ*(3 + (3 - σ)*σ)-9)-19))))/5040,
        (120 + 7*σ*(56 + σ*(72 + σ*(40 + 3*σ^2*(σ*(σ-2)-4)))))/5040,
        (1 + 7*σ*(1 + σ*(3 + σ*(5 + σ*(5 + σ*(3 + (1 - σ)*σ))))))/5040,
        σ^7/5040 ]
    else
        throw(DomainError(nw, "unspecified size"))
    end
end

# ╔═╡ ba5e2041-6014-491a-ba9d-9fcb31c0b8db
function spline_prefilter(n, nw)
    nb = (nw - 1)÷2
    A = BandedMatrix{Float64}(undef, (n,n), (nb,nb))
    if nw == 4
        A[band(0)] .= 2/3
        A[band(1)] .= A[band(-1)] .= 1/6
    elseif nw == 8
        A[band(0)] .= 151/315
        A[band(1)] .= A[band(-1)] .= 1191/5040
        A[band(2)] .= A[band(-2)] .= 120/5040
        A[band(3)] .= A[band(-3)] .= 1/5040
    else
        throw(DomainError(nw, "unspecified size"))
    end
    return A
end

# ╔═╡ 58e17999-8576-4597-96f6-46ac15ee9a8b
A4 = spline_prefilter(size(sparse,1), 4)

# ╔═╡ f2fff3d4-adbf-435b-a4de-6ce136cfa039
begin
	bplots1 = Dict{String, Plots.Plot}()
	bplots2 = Dict{String, Plots.Plot}()
	for nw in (4,8)
	    name = "B-spline $(nw-1)"
	    A = spline_prefilter(size(sparse,1), nw)
	    
	    sp1 = mapslices(x -> A \ x, sparse; dims=1)
	    in1 = mapslices(x -> int1(nw, spline, x, xdense, dx, x0), 
	                    sp1; dims=1)
	    sp2 = mapslices(x -> A \ x, in1; dims=2)
	    in2 = mapslices(x -> int1(nw, spline, x, xdense, dx, x0), 
	                    sp2; dims=2)
	    
	    error = in2 - dense
	    
	    bplots1[name] = plot(xdense, error[:,250], ylim=[-0.15,0.15],
	                   title="$name Interpolation Error", 
	                   label=:none, color=nw)
	    bplots2[name] = heatmap(xdense, xdense, error, 
	                           color=:grays, title=name)
	end
end

# ╔═╡ c71ba1e1-ebee-458b-aff7-9240f5091f7f
plot(bplots1["B-spline 3"], bplots1["B-spline 7"], layout=(2, 1))

# ╔═╡ 05bf8626-5f0c-4b86-be24-013d98db8930
plot(bplots2["B-spline 3"], bplots2["B-spline 7"], layout=(1, 2))

# ╔═╡ d3196cce-e35c-4fae-950c-22e85efd8c32
md"""
* **Unser, M., 1999, Splines: a perfect fit for signal and image processing: IEEE Signal Processing Magazine, 16, 22–38.**
* **Unser, M., A. Aldroubi, and M. Eden, 1993, B-spline signal processing: Part I – Theory: IEEE Transactions on Signal Processing, 41, 821–832.**
"""

# ╔═╡ 05331f58-c30b-430b-b7a4-df99576f6ec1
md"""
!!! assignment
    ## Task 3

    To plot spline interpolators, we must include division by $B(Z)$.
"""

# ╔═╡ 801ff6a6-2b03-45d3-91e2-9e05de7d9e75
function plot_spline_interpolator(interpolator::Function, nw::Int)
    σ = 0.7 # example shift
    w = interpolator(σ, nw)
    name = uppercasefirst(String(Symbol(interpolator)))
    # division by B(Z)
    A = spline_prefilter(64, nw)
    T = eltype(w)
    # pad with zeros and filter
    wpad = A \ vcat(zeros(T,32-nw÷2), w, zeros(T,32+nw÷2-nw))  
    # plot with stems
    p1=plot(zeros(T, 64), label=:none, color=:black)
    plot!(p1, wpad, line=:stem, marker=:circle, border=:none,
          title="$name-$nw Filter for σ=$σ", label=:none)
    # plot spectrum
    f = rfftfreq(64, 1)
    spectrum = abs.(rfft(wpad))
    p2 = plot(f, spectrum, label=:none, title="Spectrum", xlabel="frequency")
    # display side by side
    return plot(p1, p2, layout=(1, 2))
end

# ╔═╡ 1934cf47-68aa-4f0f-a27d-f2b25f1af546
md"""
Plot B-spline interpolators for filter sizes of 4 and 8.
"""

# ╔═╡ 36925ac7-4685-4ef1-8ad3-11becd704ecc
md"""
!!! assignment 
    ## Task 4

    As an alternative to using banded matrices, we can implement the division by $B(Z)$ in a signal-processing manner by factoring this filter into a product of minimum-phase filters $B(Z)=S(Z)\,S(1/Z)$ and performing recursive filtering.
"""

# ╔═╡ b9175dcb-a45e-43d0-a5af-ebee67c8e2dc
md"""
Recall functions for convolution, inverse convolution (recursive filtering), and Wilson-Burg spectral factorization.
"""

# ╔═╡ 9927d85a-d0dc-41d9-b21c-28061ef592ae
function convolve(x, b, adjoint=false)
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

# ╔═╡ 14cca030-5dff-449a-b628-af9335738c56
function recursive(x, a, adjoint=false)
    nx, na = length(x), length(a)
    y = similar(x)
    if (adjoint)
        for ix in nx:-1:1
            t = x[ix] # assume a[1]=1
            for ia in 2:min(na, nx-ix+1)
                 t -= a[ia] * y[ix+ia-1]
            end
            y[ix] = t
        end
    else
        for ix in 1:nx
            t = x[ix] # assume a[1]=1
            for ia in 2:min(na, ix)
                 t -= a[ia] * y[ix-ia+1]
            end
            y[ix] = t 
        end
    end
    return y
end

# ╔═╡ 2334eea5-f618-407a-96b1-0fd0fd7a8b03
function wilson(auto::Vector{T}, niter=10, pad=5) where T <: Real
    n = (length(auto) + 1) ÷ 2
    npad = pad*n
    nc = n + npad
    # pad with zeros
    au = vcat(zeros(T,npad), auto, zeros(T, npad))
    # initialize filter
    a = zeros(T,n)
    a[1] = 1
    b = similar(a)
    for iter in 1:niter
        bb = recursive(au, a, false)  # S/A
        cc = recursive(bb, a, true)   # S/(AA')        
        b[1] = 1
        for i in 2:n # b = Causal[1+cc]
            b[i] = 0.5*(cc[nc+i-1] + cc[nc-i+1]) / cc[nc] 
        end
        println("iter=$iter ϵ=$(maximum(b[2:end]))") 
        a = convolve(b, a) # c = A b
        b[1] = sqrt(cc[nc])
    end
    return a * b[1]
end

# ╔═╡ c22c68a0-462f-42d0-810a-fadfcda9f4e0
md"""
Recal that $B_3(Z) = Z^{-1}/6 + 2/3 + Z/6\;.$
"""

# ╔═╡ 2588af8d-1913-448c-b112-93d7d53b1831
s4 = wilson([1/6,2/3,1/6])

# ╔═╡ 53133304-e316-4d49-8f3d-3d8420251e20
function prefiilter(x, nw::Int)
    if nw==4
        filt=s4
    elseif nw==8
        filt=s8
    else
        throw(DomainError(nw, "unspecified size"))
    end
    f0=1/(filt[1]*filt[1])
    filt /= filt[1]
    
    y = recursive(x*f0, filt, true)
    return recursive(y, filt, false)
end

# ╔═╡ 7ebfc683-ea09-4608-a27a-cc012355e1f1
begin
	n = 100
	nw = 4
	
	spike = zeros(Float32, n)
	spike[(n+1)÷2] = one(Float32)
	A = spline_prefilter(n, nw)
	
	@assert A \ spike ≈ prefiilter(spike, nw)
end

# ╔═╡ 71d60d9f-0aa7-4567-afe5-2d72e3ea7261
@btime A \ spike;

# ╔═╡ 0abc9e69-914d-433f-9838-cd8d11fdc031
@btime prefiilter(spike, nw);

# ╔═╡ 239eb8ef-7833-44d5-afba-c32e2d02d546
md"""
**Your task**: Find filter `s8` corresponding to the spectral factorization of $B_7(Z)$ and test the `prefilter` program with `nw=8`.
"""

# ╔═╡ 99ef4121-9ca2-44af-b9cb-b3488d704e3e
md"""
## All-pass filters   

In B-spline interpolation, the denominator filter $B(Z)$ is independent of $\sigma$, which is convenient because we can prefilter the data by dividing by $B(Z)$ before performing interpolation.
"""

# ╔═╡ 5cb69bad-6fd0-4c4b-955b-806cbe0ab8ad
md"""
If we design an interpolation filter involving an infinite impulse response (polynomial division) with the denominator depending on $\sigma$, we can make the filter *all-pass* so that its spectrum is exactly one, similar to the ideal interpolator. This approach implies approximating the filter's phase.
"""

# ╔═╡ 46d90c16-58b2-472a-8271-ad4ad8d6a63b
md"""
The all-pass IIR approximation is given by

$$Z^{\sigma} \approx \displaystyle F(Z,\sigma) = \frac{T(Z,\sigma)}{T(1/Z,\sigma)}\;.$$

Filters $F(Z,\sigma)$ are known as *Thiran filters*. Note that the autocorrelation $F(Z,\sigma)\,F(1/Z,\sigma) = 1$, so the filter passes all frequencies (has an ideal box-shaped spectrum) while approximating linear phase.
"""

# ╔═╡ db0015f3-52f2-4dfb-816a-ed528a4c4e20
md"""
Thiran filters can be
defined using polynomials

$$T(Z,\sigma)=\sum_{k=-N}^N a_k(\sigma) Z^{-k},$$

with coefficients that are polynomials of $\sigma$:

$$\begin{array}{rcl} a_k(\sigma) & = & \displaystyle 
\frac{(2N)!(2N)!}{(4N)!(N+k)!(N-k)!} \times \\
& & \displaystyle 
\prod_{m=0}^{N-1-k}(m-2N+\sigma)
\prod_{m=0}^{N-1+k}(m-2N-\sigma)\;.\end{array}$$
"""

# ╔═╡ a89fe010-05b0-4054-b4fb-8f1044ee2f2c
md"""

In the case of $N=1$,

$$\begin{array}{rcl} T(Z,\sigma) & = & \displaystyle 
\frac{(1+\sigma)(2+\sigma)}{12}\,Z^{-1} \\
& & \displaystyle  +\frac{(2+\sigma)(2-\sigma)}{6} \\
& & \displaystyle +\frac{(1-\sigma)(2-\sigma)}{12}\,Z\;.\end{array}$$
"""

# ╔═╡ 75ada085-9b29-450a-b2e6-577e45018d6b
md"""
* **Thiran, J., 1971, Recursive digital filters with maximally flat group delay: IEEE Transactions on Circuit Theory, 18, 659–664.**
* **Zhang, X., 2009, Maxflat fractional delay IIR filter design: IEEE Transactions on Signal Processing, 57, 2950–2956.**
* **Chen, Z., S. Fomel, and W. Lu, 2013, Accelerated plane-wave destruction: Geophysics, 78, V1–V9.**
"""

# ╔═╡ 2f1b746c-e69e-4542-80c2-315d5b5025fb
md"""
## Data example

We will use a  slice out of a 3-D CT-scan of a carbonate rock sample to test different interpolation methods with a real data example.
"""

# ╔═╡ b251bb40-afe5-43e2-b51f-841658fc0162
download("https://ahay.org/data/ctscan/slice.rsf@", "slice.bin")

# ╔═╡ 13ae6ab0-5536-459f-96ca-81a922cdc24f
slice = Array{UInt8}(undef, 512, 512); # byte array

# ╔═╡ 1605acad-9b03-4737-bc33-353b266b7f49
read!("slice.bin", slice)

# ╔═╡ 1f560997-2d57-44ea-8c9e-dfaf71c0a87d
plot_scan(scan, title) = heatmap(scan, title=title, c=:grays, aspect_ratio=:equal, 
                         legend=:none, border=:none) 

# ╔═╡ f080f69f-bffe-4467-965b-c23f2c87b8f6
begin
	ps1 = plot_scan(slice, "CT scan")
	ps2 = plot_scan(reverse(transpose(slice); dims=1), "90° rotation")
	plot(ps1, ps2, layout=(1, 2))
end

# ╔═╡ 510d3634-2d93-4d02-a4ef-65eccf158bcd
md"""
We will perform a coordinate transformation on the original data. The transformation we will examine is a coordinate rotation. A 90-degree rotation is a simple transpose. However, rotating by a different angle requires interpolation from the original grid to the transformed grid.
"""

# ╔═╡ 9fa75646-f30b-4109-8cc0-f1256a3fb3b5
function rotate(image, angle, interpolator::Function, nw::Int)
    n1, n2 = size(image)
    rotated = zeros(eltype(image), (n1, n2))
    # degrees to radians
    cosa, sina = cos(π * angle/180), sin(π * angle/180)
    # central point
    c1, c2 = (n1 + 1)/2, (n2 + 1)/2    
    for i2 in 1:n2, i1 in 1:n1
        # rotated coordinates
        x1 = c1 + (i1 - c1)*cosa + (i2 - c2)*sina
        x2 = c2 - (i1 - c1)*sina + (i2 - c2)*cosa
        # nearest neighbor and interpolation filters
        k1 = floor(Int, x1); w1 = interpolator(x1-k1, nw)
        k2 = floor(Int, x2); w2 = interpolator(x2-k2, nw)
        for j1 in 1:nw, j2 in 1:nw   
            l1, l2 = k1 + j1 - nw÷2, k2 + j2 - nw÷2
            if l1 >= 1 && l1 <= n1 && l2 >= 1 && l2 <= n2
                rotated[i1,i2] += w1[j1] * w2[j2] * image[l1, l2]
            end
        end
    end         
    return rotated
end

# ╔═╡ 6e518335-8eb0-4b02-b89c-da2f75560b96
fslice = Float64.(slice);

# ╔═╡ 86ca6f73-68fb-4a7c-821a-5d289aa41aa7
begin
	pfs2 = plot_scan(rotate(fslice, 90, linear, 2), "90° rotation")
	plot(ps1, pfs2, layout=(1, 2))
end

# ╔═╡ 4f8c665e-f854-4c9b-84ed-2d3a2d4027f0
md"""
To evaluate the accuracy of different methods, we can rotate the original data in small steps and compare the results of incremental rotations up to $360^{\circ}$ with the original data. We will first try linear interpolation.
"""

# ╔═╡ 2c9678b3-feb3-468a-ab1a-ee481d7baa20
begin
	r = deepcopy(fslice)
	anim = @animate for a in 1:18
	    global r = rotate(r, 20, linear, 2)
	    plot_scan(r, "$(20a)° rotation")
	end
	gif(anim, "rotate.gif", fps = 5)
end

# ╔═╡ f0909ab1-6ce9-43e5-834c-fba2905f7cc0
error = r - fslice;

# ╔═╡ 97652ab5-7a97-4220-9ddf-0a40991bff79
heatmap(error, title="Linear Interpolation Error", aspect_ratio=:equal, 
        clim=(-12, 12), color=:grays, border=:none)

# ╔═╡ 2eba7ef0-ab9e-41c4-8318-57cf0fb0d28b
md"""
!!! assignment
    ## Task 5

    Apply a more accurate interpolation method and plot its error to compare the results.
"""

# ╔═╡ f6bf91fc-5179-422e-8a76-d98935f17a20
md"""
!!! assignment 
    ## Bonus Task

    The quadratic B-spline function is

	$$\beta_2(x) = \left\{\begin{array}{rcl} \displaystyle 0 & \mbox{for} & |x| > 3/2 \\ \displaystyle \frac{1}{8}\,\left(3-2\,|x|\right)^2 & \mbox{for} & 1/2 <|x| \le 3/2 \\ \displaystyle \frac{3}{4} - x^2 & \mbox{for} & |x| \le 1/2 \end{array}\right.$$

	Implement quadratic B-spline interpolation and use the previous test examples to verify that its error falls between those of linear interpolation and cubic B-spline.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUIExtra = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"

[compat]
BandedMatrices = "~1.7.5"
BenchmarkTools = "~1.6.0"
FFTW = "~1.10.0"
HTTP = "~1.11.0"
LaTeXStrings = "~1.4.0"
Plots = "~1.41.6"
PlutoUIExtra = "~0.1.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "f8b04b352994f07c3ab73fd3c9e47151c5360ab7"

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

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "492681bc44fac86804706ddb37da10880a2bd528"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.10.4"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BandedMatrices]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "a2c85f53ddcb15b4099da59867868bd40f005579"
uuid = "aae01518-5342-5314-be14-df237901396f"
version = "1.7.5"
weakdeps = ["SparseArrays"]

    [deps.BandedMatrices.extensions]
    BandedMatricesSparseArraysExt = "SparseArrays"

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

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

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

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
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
# ╟─00b5dbb6-8d51-4522-8ddc-0ce68d48119b
# ╟─e2539189-203d-4482-900f-21fddfba4562
# ╟─c0ede6d0-2b24-4519-b69a-30cb799b4a16
# ╟─4dcd539e-2e93-42ae-8b38-654d3a0e15f1
# ╟─fa60c4a1-81f3-4db3-b289-08ed7b52b57c
# ╠═bf8bd7d5-63b3-43af-99d8-2f79e9938a0c
# ╟─7f8a931d-367f-440c-ad4b-7bac90377fd7
# ╠═f8984287-0562-4983-b50d-9ed63a540956
# ╟─10f0c8f2-64b5-4213-83c1-4f1618d404b6
# ╟─56317420-6502-4cd9-9bb0-3072c63cf304
# ╠═52a70978-1070-4567-aa15-5a0e3b925ce4
# ╠═46be2055-cfe3-4911-8d26-4202bf4d0187
# ╠═1c6dad36-cb4d-4cea-a750-5bdcd541a283
# ╠═aedcea05-f476-447e-aa98-41bec07088ca
# ╠═dd2b203f-3ae7-4d3b-bea0-21823fc97090
# ╟─d6cd774a-8902-42bf-8320-2487a2fe47f3
# ╟─6a7adaf3-203d-43b9-9607-04acb9411cfe
# ╟─28d7148f-0ad8-44e6-9e32-84504429aa5f
# ╟─c4d0b77d-683d-478d-bdab-a7f7460d4f37
# ╟─2ec3dd22-b968-4c8d-adc1-d972e8fa304c
# ╠═c4cc29f7-d49a-4920-84cb-c23dedaf2601
# ╠═38c99849-5a4e-44a8-acd9-5f3513b6bc95
# ╠═8da05bb9-ae1a-44d4-a818-1ff29eea8560
# ╠═b0607485-0d7b-48f1-86cd-f1b8ea93587f
# ╠═366ffd53-1d78-4cd0-aba0-83033fadaffb
# ╟─5ba37030-9389-40c1-9004-5c4220cd8dd0
# ╟─7814e606-cc43-458e-90bf-51e5fd96b09f
# ╟─72042f22-222a-4f0e-a563-ab7a425c1eff
# ╠═61e6c5ff-2336-4cba-b463-c7ae6cfc3d53
# ╠═60fa6edc-0945-4dd5-8a77-b3eb5fb94180
# ╠═2534e7e6-2f81-4a7b-ad67-10378e34d8b2
# ╟─e8da4381-5433-4b8d-a8af-fdf8a709c885
# ╠═35567009-0735-4ec1-bdac-db34f48d3544
# ╠═5dd7e0ee-321b-4e7c-98d5-aaf2274ad301
# ╠═d3e60bd7-ffd9-44d4-b565-59adf21127cb
# ╠═1cf5fdb9-d174-4278-9d32-36af39c21742
# ╠═790eea28-2383-4c7c-afd8-c3ecbf540b15
# ╠═ca57cb3d-7c99-438f-bea2-646bf815955a
# ╠═86747eee-a356-45d8-820b-77d29efe2d1b
# ╠═126a54af-bfa0-47a7-a9e5-cd8ecbcfdd80
# ╠═c8c361d0-30fe-483e-92d7-2eb4e697dbc9
# ╠═55a63a5e-15e8-48ff-bc86-105bc74d631d
# ╠═12c5556a-2b22-4a81-8a73-d9cd14d5401c
# ╟─18bc745a-83f6-41e8-9f42-735079e1d7a0
# ╟─6c848cd8-06c8-44b1-8a42-465405f485b9
# ╟─c8135466-c718-456d-995e-fc8f6efe008c
# ╟─3977adb5-fcda-4df2-b2b1-aa9c029c4be0
# ╠═1df4e9a9-1d71-4b3d-938b-35c220e8cf45
# ╠═8d014cdb-80e0-4f5a-916b-1a51401aeeea
# ╠═b2d94e09-4c47-4351-be46-1cdaf9b42533
# ╠═f6d86659-a232-412d-8eb8-067b306b90d5
# ╠═c8e6830c-f05e-44d7-91ea-4e68330bd8f5
# ╠═9a54ae44-a22e-4d19-90bd-63985cdc0689
# ╟─b89ab958-279a-4614-9b9a-d15593c293db
# ╠═2aea9ad8-b3e8-4f36-88df-801277b7dd13
# ╠═33c533dd-ac97-4b71-b0c1-6734b53ce9c4
# ╠═3e708a91-dee9-4930-9ce9-b47378271f08
# ╟─a0708eb3-0fb0-4f6f-8446-6d0509645025
# ╟─df0cb1aa-12cb-4760-a6f3-3f4716df4e73
# ╟─9775af25-c525-4656-a5b6-6f8af95afcfe
# ╠═a1f6c924-0513-4ed4-8cf3-724de0e316c6
# ╠═64f89d67-bee5-431d-9c69-2fb008c054e4
# ╠═cdd48d73-174e-4ec9-aa58-8d8795594470
# ╠═e7f12c0b-c906-45aa-b85a-ea10f84c0157
# ╟─55943315-aa6f-4f57-a071-12735691ca54
# ╟─279a9b3d-3770-4139-8a1a-9a5b3a25f2d6
# ╟─cf3761ff-2063-4615-88f2-e2acc72177ed
# ╟─a4471955-bf4c-4ea1-914f-dc9f97d92eba
# ╟─c787dc4a-7fa5-4abe-b0ab-8c561bc121b0
# ╠═a07489a0-0120-4e35-b3b5-aef42e85c1e9
# ╠═77c19c98-8d79-47bb-81ae-1508b571ff20
# ╠═b65f6736-8b33-4060-af1f-12c63506fa97
# ╠═13c660d1-8847-4dfd-803e-a8c51f5dd6e0
# ╠═c1226bf1-b577-457b-9f59-7c3a0942096a
# ╠═d41688fa-870e-4d84-80a3-cc6e46faeb78
# ╟─49a774b4-3c6f-4bb3-b613-6a2224428445
# ╟─9da99ac0-6b82-41fc-a0bd-91aa83df1801
# ╟─3a1493cb-bc93-48cb-a782-372549256060
# ╠═bbdea01f-6911-46eb-af3b-cfe779832b21
# ╠═d8a7131d-0bf1-462d-8cfd-d3cc5bc5f8f4
# ╠═11b69044-f0f6-4267-9047-a01bc2805c58
# ╠═7deafb32-da00-4744-ac50-d3813cacf9e4
# ╠═1b9527d7-a303-4f2c-ad6b-5643e52f4936
# ╠═e1e12986-18b6-4a45-b738-b67fca8f706a
# ╠═518ad9cf-996d-4a16-984e-5d30404d5033
# ╠═ba5e2041-6014-491a-ba9d-9fcb31c0b8db
# ╠═58e17999-8576-4597-96f6-46ac15ee9a8b
# ╠═f2fff3d4-adbf-435b-a4de-6ce136cfa039
# ╠═c71ba1e1-ebee-458b-aff7-9240f5091f7f
# ╠═05bf8626-5f0c-4b86-be24-013d98db8930
# ╟─d3196cce-e35c-4fae-950c-22e85efd8c32
# ╟─05331f58-c30b-430b-b7a4-df99576f6ec1
# ╠═801ff6a6-2b03-45d3-91e2-9e05de7d9e75
# ╟─1934cf47-68aa-4f0f-a27d-f2b25f1af546
# ╟─36925ac7-4685-4ef1-8ad3-11becd704ecc
# ╟─b9175dcb-a45e-43d0-a5af-ebee67c8e2dc
# ╠═9927d85a-d0dc-41d9-b21c-28061ef592ae
# ╠═14cca030-5dff-449a-b628-af9335738c56
# ╠═2334eea5-f618-407a-96b1-0fd0fd7a8b03
# ╟─c22c68a0-462f-42d0-810a-fadfcda9f4e0
# ╠═2588af8d-1913-448c-b112-93d7d53b1831
# ╠═53133304-e316-4d49-8f3d-3d8420251e20
# ╠═7ebfc683-ea09-4608-a27a-cc012355e1f1
# ╠═98270df2-690f-4452-81e9-021fa678abb2
# ╠═71d60d9f-0aa7-4567-afe5-2d72e3ea7261
# ╠═0abc9e69-914d-433f-9838-cd8d11fdc031
# ╟─239eb8ef-7833-44d5-afba-c32e2d02d546
# ╟─99ef4121-9ca2-44af-b9cb-b3488d704e3e
# ╟─5cb69bad-6fd0-4c4b-955b-806cbe0ab8ad
# ╟─46d90c16-58b2-472a-8271-ad4ad8d6a63b
# ╟─db0015f3-52f2-4dfb-816a-ed528a4c4e20
# ╟─a89fe010-05b0-4054-b4fb-8f1044ee2f2c
# ╟─75ada085-9b29-450a-b2e6-577e45018d6b
# ╟─2f1b746c-e69e-4542-80c2-315d5b5025fb
# ╠═b251bb40-afe5-43e2-b51f-841658fc0162
# ╠═13ae6ab0-5536-459f-96ca-81a922cdc24f
# ╠═1605acad-9b03-4737-bc33-353b266b7f49
# ╠═1f560997-2d57-44ea-8c9e-dfaf71c0a87d
# ╠═f080f69f-bffe-4467-965b-c23f2c87b8f6
# ╟─510d3634-2d93-4d02-a4ef-65eccf158bcd
# ╠═9fa75646-f30b-4109-8cc0-f1256a3fb3b5
# ╠═6e518335-8eb0-4b02-b89c-da2f75560b96
# ╠═86ca6f73-68fb-4a7c-821a-5d289aa41aa7
# ╟─4f8c665e-f854-4c9b-84ed-2d3a2d4027f0
# ╠═2c9678b3-feb3-468a-ab1a-ee481d7baa20
# ╠═f0909ab1-6ce9-43e5-834c-fba2905f7cc0
# ╠═97652ab5-7a97-4220-9ddf-0a40991bff79
# ╟─2eba7ef0-ab9e-41c4-8318-57cf0fb0d28b
# ╟─f6bf91fc-5179-422e-8a76-d98935f17a20
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
