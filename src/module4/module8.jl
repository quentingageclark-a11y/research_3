### A Pluto.jl notebook ###
# v0.20.27

#> [frontmatter]
#> chapter = 4
#> section = 9
#> order = 9
#> image = "https://raw.githubusercontent.com/fonsp/Pluto.jl/580ab811f13d565cc81ebfa70ed36c84b125f55d/demo/plutodemo.gif"
#> title = "9 Data Analysis Notebook"
#> tags = ["module4", "track_julia", "track_material", "Pluto", "PlutoUI"]
#> layout = "layout.jlhtml"

using Markdown
using InteractiveUtils

# ╔═╡ 63fb0f24-33cd-4dd6-92b8-f790b82d64d2
begin
	using PlutoUIExtra
	TableOfContents()
end

# ╔═╡ 05232191-f55c-4a89-9b48-bd0de475ba59
using Plots

# ╔═╡ 2597e646-f3b4-4eb2-b1d1-dcdc0badaef6
using LaTeXStrings

# ╔═╡ 12039f06-6153-46cf-8b50-1fe1047df4d0
using FFTW

# ╔═╡ d2a4d85e-725f-4485-8826-5aaaab3ee1de
using CSV, DataFrames

# ╔═╡ f2a89a67-3959-43e6-a063-831ffe555688
using BenchmarkTools

# ╔═╡ 103da5bc-01b7-461a-afb8-214eff70e960
using DelimitedFiles

# ╔═╡ 6578c2e1-deaa-4016-9520-5122165f630c
import HTTP

# ╔═╡ a0c7f933-c619-41a0-ba84-726ea3a17b2d
let
    notebook_dir = @__DIR__
    toc_path = joinpath(notebook_dir, "TOC_Notebook.jl")
    
    Markdown.parse("[⬅️ Back to Table of Contents](./open?path=$(HTTP.escapeuri(toc_path)))")
end

# ╔═╡ f96d0e68-6032-4f78-8fda-4871b543048b
md"""
# Smoothing

Using tools such as digital convolution and the Fourier transform allows us to design effective filters for various tasks. In this chapter, we will focus on smoothing.
"""

# ╔═╡ b0e58b53-eba1-46eb-9819-61e115578954
md"""
## Box smoothers and triangle smoothers

*Smoothing* is a fundamental operation with many applications in data analysis. Perhaps the simplest smoother is a *box*

$$B(Z) = \displaystyle \frac{1}{N}\,\left(1+Z+Z^2+\cdots+Z^{N-1}\right) = \frac{1-Z^N}{N\,(1-Z)}.$$
"""

# ╔═╡ 9d84b045-7bac-4a1c-85b7-d6c77460e428
md"""
Division by $1-Z$ is the operation of *causal integration*. The adjoint operation corresponds to anticausal integration, or division by $1-1/Z$. To understand the program below, note that the recursion $y_n = x_n - x_{n-1}$, which corresponds to convolution by $1-Z$, is inverted by the recursion $x_n = y_n + x_{n-1}$.
"""

# ╔═╡ 42671063-8927-4335-b52e-dc3eca33f768
function causint(y::Vector{T}, adjoint::Bool) where T <: Real 
    "causal integration"
    nx = length(y)
    x = similar(y)
    if adjoint # division by 1-1/Z
		t = zero(T)
        @inbounds for i in nx:-1:1
            t += y[i]
            x[i] = t
        end
    else         # division by 1-Z
		t = zero(T)
        @inbounds for i in 1:nx
            t += y[i]
            x[i] = t
        end
    end
    return x
end

# ╔═╡ 33880df7-31b9-479e-a4c2-fbfe5e9813c6
md"""
Without the division (causal integration) step, a box filter would require $N$ operations per input point. With division, the filter needs only two additions and one multiplication (by $1/N$) per point, regardless of $N$. This speedup clearly demonstrates the power of recursive filtering.
"""

# ╔═╡ 576483ec-e266-492f-b638-2e78f8b485ad
md"""
A box filter is primitive because it treats all points within the box equally. The next useful construct is the *triangle* filter, defined as the autocorrelation of a box:
"""

# ╔═╡ cc5da4ce-c9e6-4f64-8653-57253e65ebdc
md"""
$$\begin{array}{rcl} T(Z) & = & B(Z)\,B(1/Z) \\ & = & 
\displaystyle \frac{1}{N^2}\,
\left[Z^{1-N} + 2\,Z^{2-N}+\cdots+(N-1)\,Z^{-1}\right. \\ 
& & \left. + N +(N-1)\,Z+\cdots+2\,Z^{N-2}+Z^{N-1}\right] \\
& = & \displaystyle \frac{(1-Z^N)\,(1-Z^{-N})}{N^2\,(1-Z)\,(1-1/Z)}\;.\end{array}$$
"""

# ╔═╡ 9b995790-113c-4778-8aef-184503465e1f
md"""
The numerator of the triangle filter is a three-point polynomial $(1-Z^N)\,(1-Z^{-N})=-Z^{-N}+2-Z^N$. The denominator corresponds to a chain of causal and anticausal integration, normalized by $1/N^2$.
"""

# ╔═╡ b07b4fea-0f3c-48bb-9e08-fe5649c565d9
begin
	spike = zeros(Float32, 31)
	spike[ 6] = -1
	spike[16] = 2
	spike[26] = -1
	int1 = causint(spike, false);
	int2 = causint(int1, true);
end

# ╔═╡ 83d09dba-43e3-40cb-8e85-43c6737666e4
function stems1(data, label, color) 
    plt=plot(zeros(Float32, 31), label=:none, color=:black)
    clip = 1.1*maximum(abs.(data))
    plot!(plt, data, line=:stem, marker=:circle, 
          label=label, color=color, legend=:outerleft, 
          xlim=[0.5, 31.5], ylim=[-clip, clip], border=:none)   
    return plt
end

# ╔═╡ 68af6ed8-a9bc-49c7-ac0f-c4d2238eec1c
begin
	p1 = stems1(spike, "   input", :blue);
	p2 = stems1(int1, "  B input", :blue);
	p3 = stems1(int2, "B'B input", :blue);
	plot(p1, p2, p3, layout=(3, 1))
end

# ╔═╡ ee9becd3-ce54-4c2c-9068-70ea6d3e425c
md"""
**Claerbout, J. F., 1992, Earth Soundings Analysis: Processing Versus Inversion: Blackwell Scientific Publications.**
"""

# ╔═╡ 94fb871a-fb88-42d5-a9b3-5e364c7d680b
md"""
## From triangles to Gaussians

What forms do box and triangle filters take in the Fourier domain? For simplicity, let's shift the box so it is symmetric about the origin. 

Its continuous Fourier transform is
"""

# ╔═╡ 73fb5867-ba8f-4bd4-a348-46127342d143
md"""
$$\begin{array}{rcl} F_B(\omega) & = & \displaystyle \frac{1}{2\,b}\,\int\limits_{-b}^{b}
e^{-i\omega\,t}\,dt \\ & = &
\displaystyle \frac{1}{2\,i\omega\,b}\,\left(e^{i\omega\,b}-e^{-i\omega\,b}\right) \\ & = &
\displaystyle \frac{\sin(\omega\,b)}{\omega\,b} = \mbox{sinc}(\omega\,b)\;.\end{array}$$
"""

# ╔═╡ d50e597b-df78-4686-bd2e-85e86b61462b
md"""
The sinc function reaches its maximum at zero frequency ($\omega=0$) and decays with oscillations as $1/|\omega|$ at higher frequencies. The Fourier transform of a triangular shape is

$$F_T(\omega) = \left|F_B(\omega)\right|^2 = \mbox{sinc}^2(\omega\,b)$$

and decays as $1/|\omega|^2$. 
"""

# ╔═╡ e3b13ac5-a4f3-4d2d-b5c3-7cd699e51e48
heaviside(x) = x > 0 ? 1.0 : 0.0

# ╔═╡ 70e42794-afad-4636-8ed9-1a9a739a5d80
box(x,n) =  (heaviside(x + n) - heaviside(x - n)) / (2*n)

# ╔═╡ 13aecc4c-12be-4051-a9d2-f2a01e0f2490
sinc(x,n) = sin(n*x)/(n*x)

# ╔═╡ b9ee43eb-4270-4d4b-a51a-945ed43a5d9c
begin
	pt1=plot(x -> box(x, 1), -4, 4, linewidth=2, label=:none, 
	         ylim=(-0.01, 0.51), fill=:true, fillalpha=0.1);
	pw1=plot(x -> sinc(x, 1), -25, 25, linewidth=2, label=:none, 
	         fill=:true, color=:red, fillalpha=0.1);
	pt2=plot(x -> box(x, 3), -4, 4, linewidth=2, title="Time", 
	         label=:none, xlabel=L"$t$", ylim=(-0.01,0.51), 
	         fill=:true, fillalpha=0.1);
	pw2=plot(x -> sinc(x, 3), -25, 25, linewidth=2, title="Frequency", 
	         label=:none, xlabel=L"$\omega$", color=:red, 
	         fill=:true, fillalpha=0.1);
	plot(pt1, pw1, pt2, pw2, layout=(2, 2))
end

# ╔═╡ 36dd3c50-6757-4e34-be3f-c5e599784023
md"""
If we continue to raise the sinc function to higher powers in the Fourier domain, it corresponds to repeated convolution with the box filter in the time domain. Eventually (at large $n$), the function $F_B^n(\omega)$ approaches the Gaussian.

$$\mbox{sinc}^n(\omega\,b) \approx e^{-\omega^2\,b^2\,n/6}\;.$$

This fact follows from a more general principle, the *central limit theorem*.
"""

# ╔═╡ 4c3ec481-7e93-42e9-90c0-6257e51c50f4
md"""
Suppose that a function $f(x)$ behaves at small $x$ as

$$f(x) = 1 - \alpha\,x^2 + o(x^2)\;.$$

In other words, $f(0)=1$, $f'(0)=0$, and $f''(0)=-2\alpha$. We can show that

$$\displaystyle \lim_{n \to \infty} f^n\left(\frac{x}{\sqrt{n}}\right) = e^{-\alpha\,x^2}\;.$$
"""

# ╔═╡ d7ab6f2c-5798-4bb7-af2d-c4c9c1548836
md"""
To prove this limit, consider the limit of the natural logarithm 

$$\displaystyle \lim_{n \to \infty} \ln\left[f^n\left(\frac{x}{\sqrt{n}}\right)\right] = \lim_{n \to \infty} \left(n\,\ln\left[f\left(\frac{x}{\sqrt{n}}\right)\right]\right)$$

and apply L'Hôpital's rule twice
"""

# ╔═╡ 13c1966b-2c62-4a02-8b8f-41172b752708
md"""
$$\lim_{n \to \infty} \frac{\displaystyle \ln\left[f\left(\frac{x}{\sqrt{n}}\right)\right]}{\displaystyle n^{-1}}$$
"""

# ╔═╡ 2fa1dd10-d415-4ee1-9b74-98c239c50aa9
md"""
$$= \lim_{n \to \infty} \frac{\displaystyle x\,f'\left(\frac{x}{\sqrt{n}}\right)}{\displaystyle 2\,n^{-1/2}\,f\left(\frac{x}{\sqrt{n}}\right)}$$
"""

# ╔═╡ b5ed7880-81ff-445a-b8bb-c6fc45617cc2
md"""
$$\begin{array}{rcl}
& = &
\lim_{n \to \infty} \frac{\displaystyle n^{-3/2}\,x^2\,f''\left(\frac{x}{\sqrt{n}}\right)}
{\displaystyle 2\,n^{-3/2}\,f\left(\frac{x}{\sqrt{n}}\right)+2\,n^{-2}\,x\,f'\left(\frac{x}{\sqrt{n}}\right)} \\ & = &
\displaystyle \frac{x^2}{2}\,\frac{f''(0)}{f(0)} = -\alpha\,x^2\;.\end{array}$$

Applying the exponential function to both sides yields the original limit and completes the proof of the central limit theorem.
"""

# ╔═╡ c967806e-6eef-488a-bf03-27f867909205
md"""
Returning to the function $F_B(\omega)$, we can see that it behaves at low frequencies like

$$\mbox{sinc}(\omega\,b) = 1 - \frac{b^2\,\omega^2}{3!} + O(\omega^4)$$

and therefore satisfies the central limit theorem.
"""

# ╔═╡ f48fc988-03af-4d8d-88e3-ad5b9b30d909
md"""
In the time domain, repeated convolutions with a box filter produce piecewise, compactly-supported polynomial functions of increasing smoothness (known as *B-splines*), which also converge to a Gaussian.

* **Unser, M., A. Aldroubi, and M. Eden, 1992, On the asymptotic convergence of B-spline wavelets to Gabor functions: IEEE Transactions on Information Theory, 38, 86-87.**
"""

# ╔═╡ 188f0ec4-573b-463c-8a61-2f4eb7dcdb67
md"""
To show that the inverse Fourier transform of a Gaussian is a Gaussian, consider first the integral

$$I = \int\limits_{-\infty}^{\infty} e^{-x^2}\,dx$$
"""

# ╔═╡ a4954ea9-7230-472f-a334-33019a4d9b56
md"""
and notice that

$$\begin{array}{rcl}I^2 & = & \displaystyle \int\limits_{-\infty}^{\infty}\,\int\limits_{-\infty}^{\infty} e^{-x^2-y^2}\,dx\,dy \\
    & = & 
\displaystyle  \int\limits_{0}^{2\pi}\,\int\limits_{0}^{\infty} e^{-r^2}\,r\,dr\,d\theta = \left.-2\pi\,\frac{e^{-r^2}}{2}\right|_{0}^{\infty} = \pi\end{array}$$

and, therefore, $I=\sqrt{\pi}$.
"""

# ╔═╡ cfcf9af8-9823-4bf6-80d6-3b47544ac7af
md"""
The inverse Fourier transform of a Gaussian is

$$\begin{array}{rcl}f_G(t) & = & \displaystyle \frac{1}{2\pi}\,\int\limits_{-\infty}^{\infty} e^{-\omega^2\,b^2\,n/6+i\omega\,t}\,d\omega \\
& = & \displaystyle \frac{1}{2\pi\,b\,\sqrt{n/6}}\,e^{-\frac{t^2}{2\,b^2\,n/3}}\,\int\limits_{-\infty}^{\infty} e^{-x^2}\,dx \\ & = &
    \displaystyle \frac{1}{b\,\sqrt{2\pi\,n/3}}\,e^{-\frac{t^2}{2\,b^2\,n/3}}\;.\end{array}$$
"""

# ╔═╡ 46c925f5-ed67-4a4c-806a-cdd21cdabb40
tgauss(x,a) = 1/sqrt(2π*a*a/3)*exp(-x*x/(2*a*a/3))

# ╔═╡ ebbb8a4b-17df-47f7-9ef5-13e72bb6d2f3
wgauss(x,a) = exp(-x*x*a*a/6)

# ╔═╡ 7ca235ec-ca2f-4838-8af8-12b903bb67f7
begin
	pgt1=plot(x -> tgauss(x, 2), -20, 20, linewidth=2, label=:none, 
	         ylim=(-0.01,0.36), fill=:true, fillalpha=0.1);
	pgw1=plot(x -> wgauss(x, 2), -5, 5, linewidth=2, label=:none, 
	         color=:red, fill=:true, fillalpha=0.1);
	pgt2=plot(x -> tgauss(x, 6), -20, 20, linewidth=2, title="Time", 
	         label=:none, xlabel=L"$t$", ylim=(-0.01,0.36), 
	         fill=:true, fillalpha=0.1);
	pgw2=plot(x -> wgauss(x, 6), -5 ,5, linewidth=2, title="Frequency", 
	         label=:none, xlabel=L"$\omega$", color=:red, 
	         fill=:true, fillalpha=0.1);
	plot(pgt1, pgw1, pgt2, pgw2, layout=(2, 2))
end

# ╔═╡ f249961a-45b2-48be-ad8b-dc5e11d12f5e
md"""
!!! note

    Convolution with a Gaussian in the time domain is equivalent to multiplication by a Gaussian in the frequency domain.
"""

# ╔═╡ d296428d-0f53-466e-bb2a-95672ec6abba
function doubint!(x::Vector{T}) where T <: Real
    "causal and anticausal integration in place"
    n = length(x)  
    for range in (1:n, n:-1:1)
        t = zero(T)
        @inbounds for i in range
            t += x[i]
            x[i] = t
        end
    end
end

# ╔═╡ 1e4ea1fe-d8c6-4866-a4d7-02c21aae8d5d
function triangle(x::Vector{T}, nb::Integer) where T <: Real
    "triangle filtering"
    n = length(x)
    # numerator
    t = zeros(T,n + 2*nb)
	scale = inv(T(nb*nb))
    @inbounds for i in 1:n
        xi = x[i]*scale
        t[i] -= xi
        t[i+nb] += 2xi
        t[i+2*nb] -= xi
    end
    # denominator
    doubint!(t)
    return t[1+nb:n+nb]
end

# ╔═╡ 5626b066-e226-4c54-90f5-596b1441b75c
function stems(data, n, ymax, label) 
    plt=plot(zeros(Float32, 41), label=:none, color=:black)
    plot!(plt, data, line=:stem, 
          label=:none, color=n, ylim=(0,ymax),
          xlim=[0.5, 41.5], border=:none)  
    plot!(plt, data,  ylim=(0,ymax),
          label=label, color=n, legend=:outerleft, 
          xlim=[0.5, 41.5], border=:none) 
    return plt
end

# ╔═╡ bd679a0c-46df-4960-ba29-1fcb6ed3c3db
begin
	spk = zeros(Float32, 41)
	spk[21] = 1
end

# ╔═╡ de58adbe-7beb-432e-9931-693977fda2be
md"""
To preserve the zero-frequency component of the signal, we need to ensure that the input's mean value (the zero-frequency, or *DC*, component) remains unchanged after smoothing. To implement this constraint, we can allow the output to leave the input domain and then "fold" it back into the input space.
"""

# ╔═╡ fcf9419e-2083-4009-a0cb-606a70bfd7a0
function fold(t::Vector{T}, nb::Integer) where T <: Real 
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

# ╔═╡ 14207eea-cfa9-48f4-86e6-4b1f932f7e19
function smooth(x::Vector{T}, nb::Int) where T <: Real
    "smoothing by triangle filtering with reflecting boundaries"
    n = length(x)
    t = zeros(T,n + 2*nb)
	scale = inv(T(nb*nb))
    @inbounds for i in 1:n
        xi = x[i]*scale
        t[i] -= xi
        t[i+nb] += 2xi
        t[i+2*nb] -= xi
    end
    doubint!(t)
    return fold(t, nb)
end

# ╔═╡ 75de1ca0-1bfd-42f9-92c1-22de5f1049a5
md"""
!!! assignment
    ## Task 1

	Write a function for box filtering by recursive filtering. Make sure to include reflecting boundary conditions.
"""

# ╔═╡ f417e3a7-dd0b-4b40-9fed-1c9c6ab2db5a
function boxsmooth(x::Vector{T}, nw::Integer) where T <: Real
    "smoothing by box filtering with reflecting boundaries"
    n = length(x)
    t = zeros(T,n+nw)
    for i in 1:n
        xi = x[i]/nw
        t[i] += xi
        t[i+nw] -= xi
    end
    # ADD CODE HERE TO COMPLETE THE FUNCTION
end

# ╔═╡ 97362833-ae94-4dad-8360-8a2d7b307734
begin
	spike1 = zeros(Float32, 31)
	spike1[ 6] = +1
	spike1[26] = -1
	sint1 = causint(spike1, false);
end

# ╔═╡ 05b83230-c2bf-4ec6-ace2-b4d62c2a20d3
begin
	ps1 = stems1(spike1, "  input", :blue);
	ps2 = stems1(sint1,  "B input", :blue);
	plot(ps1, ps2, layout=(2, 1))
end

# ╔═╡ 9781bb71-f87d-4d79-b6df-7fd11a5ee988
md"""
To test your program, plot the impulse response of `boxsmooth()` and compare it with that of `smooth()`.
"""

# ╔═╡ e0716f3e-98b5-4d12-a064-cb9df66072ee
md"""
!!! assignment
    ## Task 2

    Let us test box smoothing on the familiar problem of smoothing temperature data from Camp Mabry.
"""

# ╔═╡ 6637c0cc-e790-495f-a156-fdc243bb2d3f
begin
	data = DataFrame(CSV.File("mabry.csv"))
	first(data, 10)
end

# ╔═╡ 5ccf4695-d314-4cc7-bf39-83c5c04d6400
function running_mean(data, nw)
    "smoothing data in running windows of size w"
    n = length(data)
    smooth = similar(data)
    for i in 1:n
        k = max(0,min(n-nw,i-nw÷2-1))
        smooth[i] = sum(@view data[k+1:k+nw])/nw
    end
    return smooth
end

# ╔═╡ 559506bc-83e4-49c0-ab0d-f83de4572729
data.Smoothed = running_mean(data.Temperature, 30);

# ╔═╡ 2f598393-c546-417e-9deb-befae2b5de3a
begin
	plot(data.Date, data.Temperature, label="original", alpha=0.5)
	plot!(data.Date, data.Smoothed, linewidth=3, label="smoothed",
	      title="Maximum Daily Temperature at Camp Mabry",
	      ylabel="Degrees Celsius")
end

# ╔═╡ c661db95-9a1f-47d8-8381-d1f9d186ed46
md"""
**Your task**: Using the Camp Mabry data, compare the results of `running_mean` with `nw=30` and `boxfilter` with `nb=15` by both plotting the output and employing `BenchmarkTools` to measure their numerical performance.
"""

# ╔═╡ 825798c7-bb7a-4486-8bc6-4a5cf47f2b95
md"""
## Smoothing in multiple dimensions

What is the significance of the Gaussian shape? The fundamental property of the Gaussian is

$$e^{-x^2-y^2} = e^{-x^2}\,e^{-y^2}\;,$$

which shows that multidimensional Gaussian smoothing can be achieved by a simple chain of 1-D smoothing operations along different dimensions. Assessing the isotropy of multidimensional impulse responses is another way to evaluate how digital smoothing filters deviate from ideal Gaussians.
"""

# ╔═╡ eb246878-c770-4d3a-a3fd-24d012fc332f
begin
	spike2 = zeros(Float32, 1200, 400)
	spike2[300, 220] = 1
	spike2[900, 90] = 1
end

# ╔═╡ 9210a89c-dfdb-4090-9e1f-8fab3c8930a8
md"""
## Smoothing by diffusion

In physics, convolution with a Gaussian corresponds to the process of diffusion. The diffusion equation

$$\displaystyle \frac{\partial u}{\partial t} = a^2\,\nabla^2 u\;,$$

when solved in the Fourier domain, produces the solution

$$U(\mathbf{k},t) = e^{-a^2\,|\mathbf{k}|^2\,t}\,U(\mathbf{k},0)\;,$$
"""

# ╔═╡ eec9ed1b-bb40-4de9-adfd-c28b1755b7ca
md"""
where

$$U(\mathbf{k},t) = \int u(\mathbf{x},t)\,e^{-i \mathbf{k} \cdot \mathbf{x}}\,d\mathbf{x}\;.$$

Back in the spatial domain, the solution is given by the spatial convolution

$$u(\mathbf{x},t) = \displaystyle \frac{1}{(4\,\pi\,a^2\,t)^{n/2}}\,\int u_0(\mathbf{x}_0)\,\exp\left(-\frac{|\mathbf{x}-\mathbf{x}_0|^2}{4\,a^2\,t}\right)\,d\mathbf{x}_0\;,$$

where $n$ is the dimensionality of $\mathbf{x}$.
"""

# ╔═╡ b5e60c72-9de0-4bd4-9242-ecb8888ae9a6
md"""
### Discretization

Discretizing the spatial derivatives (Laplacian $-\nabla^2$) in the diffusion equation, we can write

$$\displaystyle \frac{\partial \mathbf{u}}{\partial t} = - a^2\,\mathbf{D\,u}\;,$$
"""

# ╔═╡ bc8d1b28-e4bf-4453-98e5-f0322d724b41
md"""
After the first-order discretization of the time derivative,

$$\displaystyle \frac{\mathbf{u}_{t+1} - \mathbf{u}_{t}}{\Delta t} =  - a^2\,\mathbf{D\,u}\;.$$
"""

# ╔═╡ 958c10eb-afb0-4cf8-8d43-b75350f74cda
md"""
The next step depends on whether we assign $\mathbf{u}$ on the right-hand side to $\mathbf{u}_t$ or $\mathbf{u}_{t+1}$. The first choice yields an *explicit* finite-difference scheme

$$\mathbf{u}_{t+1} = \left(\mathbf{I} - \alpha\,\mathbf{D}\right)\,\mathbf{u}_t\;,$$

where $\mathbf{I}$ is the identity operator and $\alpha = a^2\,\Delta t$. This scheme is conditionally stable and typically requires small time steps $\Delta t$.
"""

# ╔═╡ 80566129-7335-4e33-b4b2-e478b4d82924
md"""
The second choice leads to the *implicit scheme*

$$\mathbf{u}_{t+1} = \left(\mathbf{I} + \alpha\,\mathbf{D}\right)^{-1}\,\mathbf{u}_t\;,$$

which can be unconditionally stable but requires an inversion.
"""

# ╔═╡ 8329f54e-c10a-400f-8cc5-a54fbf13adee
md"""
### Implicit diffusion

In 1-D, using the simplest three-point discretization of the second derivative, the operator $\mathbf{D}$ corresponds to the filter

$$D(Z) = \displaystyle \frac{1}{(\Delta x)^2} \left(-Z^{-1} + 2 - Z\right)\;.$$

and we can consider inversion as polynomial division. 
"""

# ╔═╡ 2fc104b3-f216-4b21-bf02-2faa6bfe02e2
md"""
Factoring the symmetric filter in the denominator, smoothing via implicit diffusion becomes

$$S(Z) = \displaystyle \frac{1}{1 + \alpha\,D(Z)} = \frac{(1-\beta)^2}{(1-\beta\,Z)\,(1-\beta\,Z^{-1})}$$

with suitably chosen $\beta$.
"""

# ╔═╡ faddceab-24e4-4145-b41a-5ea78e2bb4e1
function idiffuse(x::Vector{T}, β) where T <: Real
    "Smoothing by one step of implicit diffusion"
    n = length(x)
    y = deepcopy(x)
    yi = y[1] 
    for i in 2:n-1
        y[i] = yi = (1-β)*y[i] + β*yi
    end
    y[n] = yi = y[n]/(1+β) + β*yi/(1+β)
    for i in n-1:-1:1
        y[i] = yi = (1-β)*y[i] + β*yi
    end
    return y
end

# ╔═╡ 77844a94-3f17-4dca-a933-73a945970e45
md"""
![](https://i0.wp.com/www.mines.edu/geophysics/wp-content/uploads/sites/30/2017/02/dave-hale.jpg)

"I write a lot of software, but here is my favorite ten-line program"

[https://github.com/softwareunderground/52things/blob/master/chapters/Hale.md](https://github.com/softwareunderground/52things/blob/master/chapters/Hale.md)
"""

# ╔═╡ ae836003-8029-47cb-b911-71e148179198
md"""
The recursive implicit-diffusion filter has an infinite impulse response that decays exponentially. Repeated applications of the filter converge to a Gaussian.
"""

# ╔═╡ e4da385e-0a73-4755-a8f5-efd85f76f6e3
md"""
!!! assignment
    ## Task 3

    How can we relate the parameter $\beta$ in the implicit-diffusion filter to the parameter $N$ in the triangle filter?
"""

# ╔═╡ 487c8aad-eab4-4178-a6f8-9785d06fe839
md"""
Note that Fourier response of implicit diffusion and its Taylor expansion are (assuming $\Delta t = 1$)

$$\begin{array}{rcl} S(\omega) & = & \displaystyle \frac{(1-\beta)^2}{1 - 2\beta\,\cos\omega + \beta^2} =  
\frac{1}{\displaystyle 1 + \frac{4\beta}{(1-\beta)^2}\,\sin^2\left(\frac{\omega}{2}\right)} \\
& = & \displaystyle 1 - \frac{\beta}{(1-\beta)^2}\,\omega^2 + O\left(\omega^4\right)\end{array}$$
"""

# ╔═╡ 1c2b2939-75bc-4756-a2f9-dcf8f00b48ab
md"""
The Fourier response of triangle filtering and its Taylor expansion are

$$\begin{array}{rcl} T(\omega) & = & \displaystyle \frac{1-\cos(N\omega)}{N^2\,(1-\cos{\omega})} 
= \frac{\sin^2\left(\frac{N\omega}{2}\right)}{N^2\,\sin^2\left(\frac{\omega}{2}\right)} \\
& = & \displaystyle 1 - \frac{(N^2-1)\,\omega^2}{12} + O\left(\omega^4\right)\end{array}$$

Compare the two Taylor expansions to find the relationship between $N$ and $\beta$ and express it as a function.
"""

# ╔═╡ 18509c52-f4c8-47fe-b781-ce6cbdd09886
β(nb) = 0.5 # MODIFY ME

# ╔═╡ be30de19-eec7-4b18-b9a0-b3ba04b31876
md"""
### Explicit diffusion

The key to making explicit diffusion stable is to make the time step $\Delta t$ variable.

Taking another look at the box filter 

$$B_N(Z) = \frac{\left(1-Z^{N}\right)}{N\,(1-Z)} = \frac{1}{N}\,\left(1 + Z + \cdots + Z^{N-1}\right)\;,$$

we can observe that the roots of the box filter $B_N(Z)$ are distributed along the unit circle in the complex plane:
"""

# ╔═╡ 7401278d-c731-41a5-b6f3-bb2e729a8722
md"""
$$1-Z^N = \displaystyle \prod_{n=0}^{N-1} \left(1-\frac{Z}{Z_n}\right)\;,$$

where 

$$Z_n=\exp\left(i\,\frac{2\pi\,n}{N}\right)\;.$$
"""

# ╔═╡ 45baa131-08ec-4bb9-911a-10a6c56ef3d9
md"""
Setting the radius of the box $N$ to an odd number, for simplicity, and grouping the roots into conjugate pairs $Z_n$ and $\overline{Z_n}$, we can represent this filter as

$$\begin{array}{rcl}B_{N}(Z) & = & \displaystyle \displaystyle \frac{1}{N}\,\prod_{n=1}^{N-1} \left(1-\frac{Z}{Z_n}\right)
\\ & = & \displaystyle \frac{1}{N}\,\prod_{n=1}^{(N-1)/2} \left(1-\frac{Z}{Z_n}\right)\,\left(1-\frac{Z}{\overline{Z_n}}\right) \\
& = & \displaystyle \frac{1}{N}\,\prod_{n=1}^{(N-1)/2} \left(1 -2\,\gamma_n\,Z + Z^2\right)\;,\end{array}$$
"""

# ╔═╡ 3adc443f-5cb8-4d87-896d-09b40a99fe12
md"""
where

$$\gamma_n=\cos{\left(\frac{2\pi\,n}{N}\right)}\;.$$


Thus, we represented the box filter as a cascade of three-point elementary filters. Continuing the cascade duplicates the filter's roots, yielding an analogous representation of the triangle filter

$$T_{N}(Z) = \frac{1}{N^2}\,\prod_{n=1}^{N-1} \left(Z+1/Z -2\,\gamma_n\right)\;.$$
"""

# ╔═╡ 719e6593-9df1-4f50-8e2d-f8a89f2b5071
md"""
How does it relate to diffusion? Smoothing with multiple steps of explicit diffusion becomes a cascade of filters

$$\begin{array}{rcl} E(Z) & = & \displaystyle \prod_{n=1}^{N} \left[1+\alpha\,D(Z)\right] \\ & = &
\displaystyle  \prod_{n=1}^{N} \left[1+\frac{\alpha}{(\Delta x)^2}\,\left(Z+1/Z-2\right)\right]\;.\end{array}$$
"""

# ╔═╡ 620acd8c-fe85-492b-bbb5-4a41ff33216f
md"""
Comparing it to the factorization of triangle smoothing, we see a fundamental similarity, with one key difference: instead of keeping $\alpha$ constant, triangle smoothing varies $\alpha$ with the cascade index $n$. This change ensures the stability of the filtering process.
"""

# ╔═╡ d2be36a5-f868-4a7c-a441-bea2c45ed6d8
md"""
Correspondingly, we can define triangle smoothing as a cascade of three-point filters

$$T_N(Z) = \displaystyle \prod_{n=1}^{N-1} \left[1+\alpha_n\,\left(Z+1/Z-2\right)\right]\;,$$

where

$$\alpha_n = \displaystyle \frac{1}{2\,(1-\gamma_n)} = \displaystyle \frac{1}{4\,\sin^2{\left(\frac{\pi\,n}{N}\right)}}\;.$$
"""

# ╔═╡ 2c63a8c4-a9ae-46f1-97b6-ce7aa0aaf161
function ediffuse(x::Vector{T}, nb::Int) where T <: Real
    "triangle smoothing by explicit diffusion"
    n = length(x)
    z, y = similar(x), copy(x) 
    @inbounds for ib in 1:nb-1
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

# ╔═╡ dd8475dd-4179-4466-ae8a-ee8e516e1a27
x = rand(1000);

# ╔═╡ e54a2a6c-1602-47fc-b2d6-7de545ce35d5
@btime ediffuse(x,5);

# ╔═╡ 6f443da5-adf1-4e99-9346-e347b4941613
md"""
!!! assignment
    ## Task 4

    Determine the smallest size of the smoothing radius `nb` that makes triangle smoothing implemented with recursive filtering `smooth` faster than explicit diffusion `ediffuse`.
"""

# ╔═╡ b10cfa4d-5e57-4d65-ac0e-1c617dc7157f
function stems(data, label, color) 
    plt=plot(zeros(Float32, 25), label=:none, color=:black)
    plot!(plt, data, line=:stem, marker=:circle, 
          label=label, color=color, legend=:outerleft, markersize=3,
          xlim=[0.5, 25.5], ylim=[1.1*minimum(data)-0.05,1.1*maximum(data)+0.05],border=:none)   
    return plt
end

# ╔═╡ 8dadd2c1-4c03-47eb-9d39-aa22cb25bc4c
function plot_triangle(spike)
	ptk = Array{Plots.Plot}(undef, 5)
	pfk = similar(ptk)
	tri = deepcopy(spike)
	for k in 1:5
	    tri = triangle(tri, 5)
	    ptk[k] = stems(tri, k, 0.2, "$k");
	    fspike = abs.(rfft(tri))
	    pfk[k] = stems(vcat(reverse(fspike[2:end]), fspike), k, 1, :none)
	end
	return ptk, pfk
end

# ╔═╡ 905536c7-420a-41e0-a52d-abd42c8549fc
begin
	ptk, pfk = plot_triangle(spk);
	pt = plot(ptk[1], ptk[2], ptk[3], ptk[4], ptk[5], layout=(5, 1));
	pf = plot(pfk[1], pfk[2], pfk[3], pfk[4], pfk[5], layout=(5, 1));
	plot(pt, pf, layout=(1, 2), plot_grid_title="           Time                           Frequency")
end

# ╔═╡ 5fa57ed4-92a2-4376-8720-f5fef28a0855
function plot_idiffuse(spike)
	ptk = Array{Plots.Plot}(undef,5)
	pfk = similar(ptk)
	spk = deepcopy(spike)
	for k in 1:5
    	spk = idiffuse(spk, 0.5)
    	ptk[k] = stems(spk, k, 0.35, "$k");
    	fspike = abs.(rfft(spk))
    	pfk[k] = stems(vcat(reverse(fspike[2:end]), fspike), k, 1.01, :none)
	end
	return ptk, pfk
end

# ╔═╡ 997512b2-2cf9-4d76-8552-afc44c7f5033
begin
	pdt, pdf = plot_idiffuse(spk)
	pit = plot(pdt[1], pdt[2], pdt[3], pdt[4], pdt[5], layout=(5, 1));
	pif = plot(pdf[1], pdf[2], pdf[3], pdf[4], pdf[5], layout=(5, 1));
	plot(pit, pif, layout=(1, 2), plot_grid_title="           Time                           Frequency")
end

# ╔═╡ ec6d786b-4af4-4005-8b84-7c985a40add7
function plot_ediffuse(x::Vector{T}, nb::Int) where T <: Real
    n = length(x)
    z, y = similar(x), copy(x) 
    plots = Array{Plots.Plot}(undef, nb)
    plots[1] = stems(y, :none, 1)
    for ib in 1:nb-1
        α = 1/(4*sin(π*ib/nb)^2)
        # reflecting boundary conditions
        z[1] = (1 - 2*α)*y[1] + α*(y[1] + y[2])
        for i in 2:n-1
            z[i] = (1 - 2*α)*y[i] + α*(y[i-1] + y[i+1])
        end
        z[n] = (1 - 2*α)*y[n] + α*(y[n-1] + y[n])
        y, z = z, y
        plots[ib+1] = stems(y, "$ib", ib+1)
    end
    return plots
end

# ╔═╡ da6db69f-281e-469c-b51e-f5c4d199b0a5
begin
	spike3 = zeros(Float32, 25)
	spike3[13] = 1
end

# ╔═╡ d680858b-822f-4caf-87ab-a20414bd565c
begin
	pek = plot_ediffuse(spike3, 7);
	pe1 = plot(pek[1], pek[2], pek[3], pek[4], layout=(4, 1))
	pe2 = plot(pek[5], pek[6], pek[7], layout=(4, 1))
	plot(pe1, pe2, layout=(1, 2))
end

# ╔═╡ 975b4498-5070-469e-8b14-cccebbd55db2
md"""
**S. Fomel, 2022, Shaping regularization by fast explicit diffusion: Second International Meeting for Applied Geoscience & Energy, 3156-3160.**
"""

# ╔═╡ 25b6b5da-d02a-4a71-8dea-589fa87c11e5
md"""
## Data application

For the last task, we return to the time slice from a seismic volume.
"""

# ╔═╡ c30e00ec-aa79-4fd6-9992-a66afa75bdea
begin
	download("https://ahay.org/data/hall/horizon.asc","horizon.asc")
	xyz = readdlm("horizon.asc") # read data from a text file
end

# ╔═╡ 1c755050-d17d-4e56-8268-5fa3476c8c30
begin
	n1, n2 = 196, 291
	iline = xyz[1:n1:end, 1]
	xline = xyz[1:n1, 2]
	horizon=reshape(xyz[:, 3], (n1, n2))
end

# ╔═╡ d5d7ee29-d6c8-4dbc-9bd4-c5b75baf6c17
heatmap(iline, xline, horizon, xlabel=L"$x$ (m)", ylabel=L"$y$ (m)", 
		title="Seismic Horizon")

# ╔═╡ 1f12fbdf-9eff-45c7-a73b-952db8987323
slice = horizon[98,:]

# ╔═╡ 243ef7c7-b5a5-49f1-a715-56419b6885d6
begin
	plot(iline, slice, linewidth=2, ylim=(55,85), clim=(55,85), xlabel=L"$x$ (m)", title="Horizontal Slice", line_z=slice, label=:none)
	annotate!(35500,73,"Channel")
	annotate!(36750,79,"Channel")
end

# ╔═╡ 672a84e9-c5e3-4429-bdb2-ad15dd6477da
md"""
* **Hall, M., 2014. Smoothing surfaces and attributes. The Leading Edge, 33(2), pp.128-129.**
"""

# ╔═╡ 2b72e66a-9ae9-4595-aa05-93b6eba0d66b
md"""
!!! assignment
    ## Task 5

    To emphasize the channel structure, it helps to remove minor fluctuations from the image. To do this, apply two-dimensional smoothing (in both $x$ and $y$). You can select the appropriate smoothing program and its parameters.
"""

# ╔═╡ 8df2f9a2-7cd2-4bfb-8efb-d3bb36eea307
md"""
!!! assignment
    ## Bonus Task

    Our triangle smoothing programs assume that the smoothing radius `nb` is an integer. To increase precision, we can extend the method to accommodate a real-valued radius. The analysis of the Taylor expansion around zero frequency suggests implementing this by interpolating between two triangles as follows:

    $$T_R(Z) = \displaystyle \frac{(N+1)^2-R^2}{2\,N+1}\,T_N(Z) + \frac{R^2-N^2}{2\,N+1}\,T_{N+1}(Z)\;,$$

    where $N$ is the largest integer smaller than $R$ so that $N \le R < N+1$.

    Write a function to implement triangle smoothing with a real-valued radius.
"""

# ╔═╡ c2a320d4-337c-493a-b970-939e076cad4f
function smooth(x::Vector{T}, R::T) where T <: Real
    "smoothing by triangle filtering with reflecting boundaries"
    nb = floor(Int, R) # largest integer smaller than R
    # !!! MODIFY BELOW !!!
    return smooth(x,nb) 
end

# ╔═╡ 67a5d16a-2473-4aed-b52f-c769089ee587
function plot_trianglebc(k1)
	n = length(k1)
	pk = Array{Plots.Plot}(undef, 3)
	hk = Array{Float32}(undef, 3)
	blank = zeros(Float32, 40)
	for k in 1:3
    	spike = deepcopy(blank)
    	spike[k1[k]] = 1
    	spike = smooth(spike, 12)
    	smax = maximum(spike)
    	plt = plot(blank, label=:none, color=:black)
    	plot!(plt, spike, line=:stem, label=:none, color=:blue, 
        	  xlim=[0.5, 40.5], ylim=[-0.05, smax+0.05], 
          	  marker=:circle, border=:none) 
    	hk[k] = smax
    	pk[k] = plt
	end
	return pk, hk
end

# ╔═╡ e4d6a4e1-5d1a-4c46-a446-01ff6278ba42
begin
	pk, hk = plot_trianglebc([17,9,1])
	plot(pk[1], pk[2], pk[3], 
	     layout=grid(3, 1, heights=hk/sum(hk)), 
	     margin=0.001*Plots.mm)
end

# ╔═╡ 7a2b1ef5-941b-42da-83b1-8d2655b2996e
function plot_triangle2(spike)
	pk = Array{Plots.Plot}(undef,3)
	spike2 = deepcopy(spike)
	for k in 1:3
    	spike2 = mapslices(col -> smooth(col, 200), spike2; dims=1)
    	spike2 = mapslices(row -> smooth(row, 100), spike2; dims=2)
    	surf = surface(spike2, legend=:none, title="smooth $k")
    	cont = contour(spike2, legend=:none)
    	pk[k] = plot(surf, cont, layout=(2, 1), axis=nothing)
	end
	return pk
end

# ╔═╡ 077cf971-1083-4d25-92d2-9701a18b6f0c
begin
	pk2 = plot_triangle2(spike2)
	plot(pk2[1], pk2[2], pk2[3], layout=(1, 3))
end

# ╔═╡ 30665a6c-c28b-4bcb-8238-a090fc62be48
@assert smooth(x,5) ≈ ediffuse(x,5)

# ╔═╡ 350f0889-ba7b-4e67-898c-940ed1ca2e84
@btime smooth(x,5);

# ╔═╡ 35c514f7-52ff-43ee-902c-41960851a614
md"""
Test your function by plotting impulse responses and applying the program to data examples from previous tasks.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUIExtra = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"

[compat]
BenchmarkTools = "~1.5.0"
CSV = "~0.10.16"
DataFrames = "~1.8.2"
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
project_hash = "a40a52ecc041bf7f17bc39fe3e576c4a2d7951da"

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

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "8d8e0b0f350b8e1c91420b5e64e5de774c2f0f4d"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.16"

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

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

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

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "0716e01c3b40413de5dedbc9c5c69f27cddfddfc"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.3"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

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
# ╟─63fb0f24-33cd-4dd6-92b8-f790b82d64d2
# ╟─6578c2e1-deaa-4016-9520-5122165f630c
# ╟─a0c7f933-c619-41a0-ba84-726ea3a17b2d
# ╟─f96d0e68-6032-4f78-8fda-4871b543048b
# ╟─b0e58b53-eba1-46eb-9819-61e115578954
# ╟─9d84b045-7bac-4a1c-85b7-d6c77460e428
# ╠═42671063-8927-4335-b52e-dc3eca33f768
# ╟─33880df7-31b9-479e-a4c2-fbfe5e9813c6
# ╟─576483ec-e266-492f-b638-2e78f8b485ad
# ╟─cc5da4ce-c9e6-4f64-8653-57253e65ebdc
# ╟─9b995790-113c-4778-8aef-184503465e1f
# ╠═b07b4fea-0f3c-48bb-9e08-fe5649c565d9
# ╠═05232191-f55c-4a89-9b48-bd0de475ba59
# ╠═83d09dba-43e3-40cb-8e85-43c6737666e4
# ╠═68af6ed8-a9bc-49c7-ac0f-c4d2238eec1c
# ╟─ee9becd3-ce54-4c2c-9068-70ea6d3e425c
# ╟─94fb871a-fb88-42d5-a9b3-5e364c7d680b
# ╟─73fb5867-ba8f-4bd4-a348-46127342d143
# ╟─d50e597b-df78-4686-bd2e-85e86b61462b
# ╠═e3b13ac5-a4f3-4d2d-b5c3-7cd699e51e48
# ╠═70e42794-afad-4636-8ed9-1a9a739a5d80
# ╠═13aecc4c-12be-4051-a9d2-f2a01e0f2490
# ╠═2597e646-f3b4-4eb2-b1d1-dcdc0badaef6
# ╠═b9ee43eb-4270-4d4b-a51a-945ed43a5d9c
# ╟─36dd3c50-6757-4e34-be3f-c5e599784023
# ╟─4c3ec481-7e93-42e9-90c0-6257e51c50f4
# ╟─d7ab6f2c-5798-4bb7-af2d-c4c9c1548836
# ╟─13c1966b-2c62-4a02-8b8f-41172b752708
# ╟─2fa1dd10-d415-4ee1-9b74-98c239c50aa9
# ╟─b5ed7880-81ff-445a-b8bb-c6fc45617cc2
# ╟─c967806e-6eef-488a-bf03-27f867909205
# ╟─f48fc988-03af-4d8d-88e3-ad5b9b30d909
# ╟─188f0ec4-573b-463c-8a61-2f4eb7dcdb67
# ╟─a4954ea9-7230-472f-a334-33019a4d9b56
# ╟─cfcf9af8-9823-4bf6-80d6-3b47544ac7af
# ╠═46c925f5-ed67-4a4c-806a-cdd21cdabb40
# ╠═ebbb8a4b-17df-47f7-9ef5-13e72bb6d2f3
# ╠═7ca235ec-ca2f-4838-8af8-12b903bb67f7
# ╟─f249961a-45b2-48be-ad8b-dc5e11d12f5e
# ╠═d296428d-0f53-466e-bb2a-95672ec6abba
# ╠═1e4ea1fe-d8c6-4866-a4d7-02c21aae8d5d
# ╠═5626b066-e226-4c54-90f5-596b1441b75c
# ╠═12039f06-6153-46cf-8b50-1fe1047df4d0
# ╠═bd679a0c-46df-4960-ba29-1fcb6ed3c3db
# ╠═8dadd2c1-4c03-47eb-9d39-aa22cb25bc4c
# ╠═905536c7-420a-41e0-a52d-abd42c8549fc
# ╟─de58adbe-7beb-432e-9931-693977fda2be
# ╠═fcf9419e-2083-4009-a0cb-606a70bfd7a0
# ╠═14207eea-cfa9-48f4-86e6-4b1f932f7e19
# ╠═67a5d16a-2473-4aed-b52f-c769089ee587
# ╠═e4d6a4e1-5d1a-4c46-a446-01ff6278ba42
# ╟─75de1ca0-1bfd-42f9-92c1-22de5f1049a5
# ╠═f417e3a7-dd0b-4b40-9fed-1c9c6ab2db5a
# ╠═97362833-ae94-4dad-8360-8a2d7b307734
# ╠═05b83230-c2bf-4ec6-ace2-b4d62c2a20d3
# ╟─9781bb71-f87d-4d79-b6df-7fd11a5ee988
# ╟─e0716f3e-98b5-4d12-a064-cb9df66072ee
# ╠═d2a4d85e-725f-4485-8826-5aaaab3ee1de
# ╠═6637c0cc-e790-495f-a156-fdc243bb2d3f
# ╠═5ccf4695-d314-4cc7-bf39-83c5c04d6400
# ╠═559506bc-83e4-49c0-ab0d-f83de4572729
# ╠═2f598393-c546-417e-9deb-befae2b5de3a
# ╟─c661db95-9a1f-47d8-8381-d1f9d186ed46
# ╟─825798c7-bb7a-4486-8bc6-4a5cf47f2b95
# ╠═eb246878-c770-4d3a-a3fd-24d012fc332f
# ╠═7a2b1ef5-941b-42da-83b1-8d2655b2996e
# ╠═077cf971-1083-4d25-92d2-9701a18b6f0c
# ╟─9210a89c-dfdb-4090-9e1f-8fab3c8930a8
# ╟─eec9ed1b-bb40-4de9-adfd-c28b1755b7ca
# ╟─b5e60c72-9de0-4bd4-9242-ecb8888ae9a6
# ╟─bc8d1b28-e4bf-4453-98e5-f0322d724b41
# ╟─958c10eb-afb0-4cf8-8d43-b75350f74cda
# ╟─80566129-7335-4e33-b4b2-e478b4d82924
# ╟─8329f54e-c10a-400f-8cc5-a54fbf13adee
# ╟─2fc104b3-f216-4b21-bf02-2faa6bfe02e2
# ╠═faddceab-24e4-4145-b41a-5ea78e2bb4e1
# ╟─77844a94-3f17-4dca-a933-73a945970e45
# ╠═5fa57ed4-92a2-4376-8720-f5fef28a0855
# ╠═997512b2-2cf9-4d76-8552-afc44c7f5033
# ╟─ae836003-8029-47cb-b911-71e148179198
# ╟─e4da385e-0a73-4755-a8f5-efd85f76f6e3
# ╟─487c8aad-eab4-4178-a6f8-9785d06fe839
# ╟─1c2b2939-75bc-4756-a2f9-dcf8f00b48ab
# ╠═18509c52-f4c8-47fe-b781-ce6cbdd09886
# ╟─be30de19-eec7-4b18-b9a0-b3ba04b31876
# ╟─7401278d-c731-41a5-b6f3-bb2e729a8722
# ╟─45baa131-08ec-4bb9-911a-10a6c56ef3d9
# ╟─3adc443f-5cb8-4d87-896d-09b40a99fe12
# ╟─719e6593-9df1-4f50-8e2d-f8a89f2b5071
# ╟─620acd8c-fe85-492b-bbb5-4a41ff33216f
# ╟─d2be36a5-f868-4a7c-a441-bea2c45ed6d8
# ╠═2c63a8c4-a9ae-46f1-97b6-ce7aa0aaf161
# ╠═dd8475dd-4179-4466-ae8a-ee8e516e1a27
# ╠═30665a6c-c28b-4bcb-8238-a090fc62be48
# ╠═f2a89a67-3959-43e6-a063-831ffe555688
# ╠═350f0889-ba7b-4e67-898c-940ed1ca2e84
# ╠═e54a2a6c-1602-47fc-b2d6-7de545ce35d5
# ╟─6f443da5-adf1-4e99-9346-e347b4941613
# ╠═b10cfa4d-5e57-4d65-ac0e-1c617dc7157f
# ╠═ec6d786b-4af4-4005-8b84-7c985a40add7
# ╠═da6db69f-281e-469c-b51e-f5c4d199b0a5
# ╠═d680858b-822f-4caf-87ab-a20414bd565c
# ╟─975b4498-5070-469e-8b14-cccebbd55db2
# ╟─25b6b5da-d02a-4a71-8dea-589fa87c11e5
# ╠═103da5bc-01b7-461a-afb8-214eff70e960
# ╠═c30e00ec-aa79-4fd6-9992-a66afa75bdea
# ╠═1c755050-d17d-4e56-8268-5fa3476c8c30
# ╠═d5d7ee29-d6c8-4dbc-9bd4-c5b75baf6c17
# ╠═1f12fbdf-9eff-45c7-a73b-952db8987323
# ╠═243ef7c7-b5a5-49f1-a715-56419b6885d6
# ╟─672a84e9-c5e3-4429-bdb2-ad15dd6477da
# ╟─2b72e66a-9ae9-4595-aa05-93b6eba0d66b
# ╟─8df2f9a2-7cd2-4bfb-8efb-d3bb36eea307
# ╠═c2a320d4-337c-493a-b970-939e076cad4f
# ╟─35c514f7-52ff-43ee-902c-41960851a614
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
