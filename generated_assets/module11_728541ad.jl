### A Pluto.jl notebook ###
# v0.20.27

#> [frontmatter]
#> chapter = 4
#> section = 12
#> order = 12
#> image = "https://raw.githubusercontent.com/fonsp/Pluto.jl/580ab811f13d565cc81ebfa70ed36c84b125f55d/demo/plutodemo.gif"
#> title = "Fundamentals of linear estimation"
#> tags = ["module4", "track_julia", "track_material", "Pluto", "PlutoUI"]
#> layout = "layout.jlhtml"

using Markdown
using InteractiveUtils

# ╔═╡ e5d0c763-9d85-4154-8a4a-df0c95d54d28
begin
	using PlutoUIExtra
	TableOfContents()
end

# ╔═╡ d1cefa45-272a-40cc-9b8c-f070e7cb1e71
using Plots

# ╔═╡ a42162f8-4c1f-492f-98e4-1590b1a915bc
using CSV, DataFrames

# ╔═╡ 39e2a830-fd2b-4ebf-81f4-f65cf4556fa8
using Dates, Statistics

# ╔═╡ 233e8fad-edf2-45bc-9a7a-6de5bafc902b
using DelimitedFiles

# ╔═╡ e63ea941-0715-4301-89bc-a6d9bed91065
using TableScraper

# ╔═╡ 9c2b8cc5-0f43-4d12-8261-24cff9361204
import HTTP

# ╔═╡ 3252a716-33bd-42e9-a3ec-8ccb4111de24
let
    notebook_dir = @__DIR__
    toc_path = joinpath(notebook_dir, "TOC_Notebook.jl")
    
    Markdown.parse("[⬅️ Back to Table of Contents](./open?path=$(HTTP.escapeuri(toc_path)))")
end

# ╔═╡ 235cea6e-05a8-4132-bf79-ffb3c82a550f
md"""
# Fundamentals of linear estimation

Data analysis is fundamentally a problem of statistical estimation: we aim to recover missing information from the available data. The process of recovery can be explained using basic statistical principles.
"""

# ╔═╡ 679977f5-7e27-481b-8584-55b711539e08
md"""
## Gauss-Markov theory

To introduce the general theory of linear estimation, let us first define its key component: *covariance* matrices. We will denote the covariance matrix between two signals $\mathbf{a}$ and $\mathbf{b}$ as $\mathbf{C}_{ab}$. By definition, the covariance of two zero-mean signals is the mathematical expectation

$$\mathbf{C}_{ab}=E\left[\mathbf{a}\,\mathbf{b}^T\right]\;,$$

where $\mathbf{b}^T$ denotes the transpose of vector $\mathbf{b}$ (a row vector). For brevity, we will use $\mathbf{C}_{a}$ to denote auto-covariance $\mathbf{C}_{aa}$. Auto-covariance is a symmetric matrix.
"""

# ╔═╡ a3a7f340-c8ff-4d1a-89d5-90a42d627994
md"""
Suppose that we measure some data $\mathbf{d}$ and want to estimate the model $\mathbf{m}$. Our estimate, $\widehat{\mathbf{m}}$ will be generally different from the actual model with the error $\mathbf{e}=\widehat{\mathbf{m}}-\mathbf{m}$.
"""

# ╔═╡ 723a5475-649b-4e8b-9471-027ad86fad65
md"""
Suppose that we look for a linear estimate

$$\widehat{\mathbf{m}}=\mathbf{L}\,\mathbf{d}$$

and let us evaluate the covariance of the error. 
"""

# ╔═╡ 8777da1d-d9cc-448c-924d-5c87e4b531b0
md"""
Using simple algebraic manipulations and assuming that $\mathbf{C}_{d}$ is invertible, we find that

$$\begin{array}{rcl}\mathbf{C}_{e} & = & E\left[(\mathbf{L}\,\mathbf{d}-\mathbf{m})\,(\mathbf{d}^T\,\mathbf{L}^T-\mathbf{m}^T)\right] \\
& = & \mathbf{L}\,\mathbf{C}_{d}\,\mathbf{L}^T -
\mathbf{L}\,\mathbf{C}_{md}^T -
\mathbf{C}_{md}\,\mathbf{L}^T+\mathbf{C}_m \\ & = &
\left(\mathbf{L}-\mathbf{C}_{md}\,\mathbf{C}_d^{-1}\right)\,\mathbf{C}_d\,\left(\mathbf{L}-\mathbf{C}_{md}\,\mathbf{C}_d^{-1}\right)^T \\
& & - \mathbf{C}_{md}\,\mathbf{C}_d^{-1}\,\mathbf{C}_{md}^T +
\mathbf{C}_m\;.\end{array}$$

Note that the second and the third terms in the last expression do not depend on $\mathbf{L}$. 
"""

# ╔═╡ be14b720-b4eb-4a32-b5ec-a2e404a5cc92
md"""
It follows that the covariance of the error is minimized when $\mathbf{L}$ is chosen as

$$\boxed{
\mathbf{L}=\mathbf{C}_{md}\,\mathbf{C}_d^{-1}
}\;.$$

This equation is the fundamental principle of linear estimation theory.

To put it in words, to make a minimum-variance linear estimate of a model, it is necessary to remove the interdependence of different data values (multiplying by $\mathbf{C}_d^{-1}$) and then to impose the dependence between model and data (multiplying by $\mathbf{C}_{md}$).
"""

# ╔═╡ cec6d5ba-638f-4047-ba11-aad35b55774c
md"""
The boxed equation is known as the *Gauss-Markov* equation and is immediately useful in many applications. As long as we have access to the relevant covariances and can efficiently invert the data covariance, the Gauss-Markov equation can be used to make predictions of unknown data.

* **Liebelt, P. B., 1967, An introduction to optimal estimation: Addison-Wesley.**
* **Daley, R., 1993, Atmospheric data analysis: Cambridge University Press.**
"""

# ╔═╡ 48fc15f0-9aa2-432a-9ddc-f6b9877b7df5
md"""
## Data example: 1997 Spatial Interpolation Contest
"""

# ╔═╡ bbe25e38-95f1-4ed9-8415-d92ca62592a3
import GMT

# ╔═╡ 40394b16-670b-42c8-8c84-7db80a60a7da
md"""
![](https://www.generic-mapping-tools.org/_static/gmt-logo.png)

[https://www.generic-mapping-tools.org/](https://www.generic-mapping-tools.org/)
"""

# ╔═╡ c2579001-ce17-419a-84f5-f2e98db93e8b
GMT.coast(proj=:Mercator, DCW=(country="CH"), 
          title="Switzerland", show=true)

# ╔═╡ 77773c15-4e89-4d96-9911-39703a51efd6
md"""
In 1997, the European Communities conducted a Spatial Interpolation Comparison. Many organizations participated, and the results were published in a special issue of the *Journal of Geographic Information and Decision Analysis* and in a separate report.
"""

# ╔═╡ acbb8073-2fc0-411f-84cb-be9523844829
md"""
* **Dubois, G., 1999, Spatial interpolation comparison 97: Foreword and introduction: Journal of Geographic Information and Decision Analysis, 2, 1–10.**
* **Dubois, G., J. Malczewski, and M. D. Cort, eds., 2003, Mapping radioactivity in the environment. Spatial Interpolation Comparison 1997.: Office for Official Publications of the European Communities.**
"""

# ╔═╡ 35b24d3b-1c01-44d4-b681-8768ce054240
# download a data file
download("https://ahay.org/data/rain/border.rsf@","border.bin")

# ╔═╡ cd323d29-43ca-4ce7-aa4b-1110bbfc982d
border = Array{Float32}(undef, 2, 1289); # single-precision array

# ╔═╡ fc18f856-5fd6-4f43-9a21-f67ff71a07e2
# read data
read!("border.bin", border)

# ╔═╡ 16a4a86e-4cc8-4f2d-9a3d-76d58fdc4fa2
# download a data file
download("https://ahay.org/data/rain/alldata.rsf@","alldata.bin")

# ╔═╡ 590e20ad-283c-4114-a4fe-15c8dbe76c8e
alldata = Array{Float32}(undef, 3, 467); # single-precision array

# ╔═╡ 356dd92d-bf99-48aa-b78f-ea33c1bf0e74
read!("alldata.bin", alldata)

# ╔═╡ f5ea975d-c191-4159-aebe-1f900238ac43
# download a data file
download("https://ahay.org/data/rain/obsdata.rsf@","obsdata.bin")

# ╔═╡ 6a581b1f-8ed8-4713-b071-9ef13cfd811d
obsdata = Array{Float32}(undef, 3, 100); # single-precision array

# ╔═╡ c4df9fab-37f0-42d4-b96c-f029995b516b
# read data
read!("obsdata.bin", obsdata)

# ╔═╡ 12c0a2c3-1938-4ddc-b9f3-86950e3bf1c5
begin
	plot(border[1,:], border[2,:], linewidth=2, label=:none)
	scatter!(alldata[1,:], alldata[2,:], ms=2, ma=0.5, label="all stations")
	scatter!(obsdata[1,:], obsdata[2,:], markershape=:utriangle, ms=4,
	    label="test stations", title="Switzerland Weather Stations")
end

# ╔═╡ 7b82027b-db18-4d3d-bcd4-9ee4c595d945
md"""
The comparison used a dataset from rainfall measurements in Switzerland on May 8, 1986, the day of the Chernobyl disaster. A total of 467 rainfall measurements were recorded that day. A randomly selected subset of 100 measurements served as the input data in the 1997 Spatial Interpolation Comparison to interpolate other measurements using different techniques and to compare the results with the known data.
"""

# ╔═╡ 35749e12-e4e1-4f5f-b949-b2f71cf6ebd2
md"""
We will approach the interpolation problem using *radial basis functions*. Let the rainfall measurements be given by

$$r(\mathbf{x}) = \displaystyle \sum\limits_{k=1}^N c_k\,f\left(\left|\mathbf{x} -\mathbf{x}_k\right|\right)\;,$$

where $\mathbf{x}_k$ are known locations, and $f(r)$ is a radial basis function, which plays the role of characterizing the covariance of the spatial distribution as a function of the distance between different points on the map.
"""

# ╔═╡ f3aa8430-2664-4ec5-9844-0c7e0de84321
md"""
Naturally, our model should satisfy the condition

$$r_j = r(\mathbf{x}_j) = \displaystyle \sum\limits_{k=1}^N c_k\,f\left(\left|\mathbf{x_j}-\mathbf{x}_k\right|\right)$$

for $j=1,2,\ldots,N$, which gives us a set of $N$ linear equations for funding the coefficients $c_k$. Solving this system of equations corresponds to inverting the data covariance matrix $\mathbf{C}_d$. After finding the coefficients, we can make predictions for rainfall at other locations, which corresponds to multiplication by $\mathbf{C}_{md}$.
"""

# ╔═╡ bd90f64c-7ecf-45e5-9d37-36d90703e58e
md"""
**Fornberg, B., and N. Flyer, 2015, A Primer on Radial Basis Functions with Applications to the Geosciences: Society for Industrial and Applied Mathematics.**
"""

# ╔═╡ 74bf6172-aae1-40fa-a00a-f99e4d3014a6
md"""
For the apriori-defined radial basis function, we will use a Gaussian with a specified width $r_0$:

$$f(r) = e^{-r^2/r_0^2}\;.$$
"""

# ╔═╡ dd34aa6d-5673-41ac-9e6f-ac60a3cf0978
# characteristic distance
r0 =  25

# ╔═╡ aabe9f28-72bc-4d49-8884-b3831371cd51
# radial basis function
f(x,y)=exp(-(x^2+y^2)/r0^2)

# ╔═╡ 4be29b22-b5d4-4dad-977d-6c3d581011db
begin
	x, y = obsdata[1,:], obsdata[2,:]
	# data covariance
	C = f.(x .- x', y .- y')
end

# ╔═╡ c356de4c-1c4f-4961-95d0-a56bf7f82404
heatmap(C, title="Data Covariance Matrix", xlim=[1,100], aspect_ratio=:equal)

# ╔═╡ 1fc47b85-cd2f-42d0-af66-c0440a2ded00
coeff = C \ obsdata[3,:];

# ╔═╡ 402cd0a6-ec38-48d3-8177-b73975a48ab2
function rain(x1, x2)
    # rainfall prediction using RBF
    r = 0.0
    for k in eachindex(x)
        r += coeff[k]*f(x1-x[k], x2-y[k])
    end
    return max(r, 0.0)
end

# ╔═╡ 9776b151-6f85-404c-a19b-b4312748b6fc
begin
	lat = -185:185
	lon = -127:127
	rs = [rain(x1,x2) for x1 in lon, x2 in lat]
end

# ╔═╡ 13a8c867-7dd7-4f12-b63a-63e871dcbf40
heatmap(lat, lon, rs, title="Predicted Rainfall", cmap=:viridis)

# ╔═╡ 14f59e33-fd48-4711-a1c4-674610781080
exact = alldata[3,:];

# ╔═╡ e0ef5c6e-489b-4ee8-afd1-6341df9b8911
pred = [rain(alldata[1,k], alldata[2,k]) for k in 1:size(alldata,2)];

# ╔═╡ 8e77a9f1-ebee-4578-b4d4-4b8c268a8259
begin
	lim = [-10,750]
	scatter(exact, pred, xlabel="True", ylabel="Predicted", 
	    aspect_ratio=:equal, xlim=lim, ylim=lim, label=:none)
	plot!(lim, lim, label=:none)
end

# ╔═╡ 616684c7-d082-4d03-9f2d-3c5d3f8ec1d4
# correlation coefficient
corr = sum(exact .* pred)/sqrt(sum(exact .* exact)*sum(pred .* pred))

# ╔═╡ 26a14a97-62fe-41cf-9e95-36f509dd4b58
md"""
!!! assignment
    ## Task 1

    Try to improve the accuracy of the prediction. You can do this by either adjusting the parameter `r0` ($r_0$ in the equation) or modifying the radial basis function. For example, the exponential basis function is

    $$f(r) = e^{-r/r_0}\;.$$
"""

# ╔═╡ 3d81afc6-f055-4262-84b9-49baf846339c
md"""
## Linear estimation

We have not said anything yet about the true dependence between $\mathbf{m}$ and $\mathbf{d}$. Let us assume that this dependence is linear:

$$\mathbf{d}=\mathbf{F}\,\mathbf{m}+\mathbf{n}\;,$$

where $\mathbf{F}$ is the linear forward-modeling operator, and
$\mathbf{n}$ is some additive noise in the data. 
"""

# ╔═╡ 6176ef7f-17fc-4f93-bcbf-2cd230f77b1a
md"""
We can additionally assume that the noise is uncorrelated with the model or, in other words, $\mathbf{C}_{mn}=0$. With that assumption, the covariances transform as follows:

$$\begin{array}{rcl} \mathbf{C}_d & = & E\left[(\mathbf{F}\,\mathbf{m}+\mathbf{n})\,(\mathbf{m}^T\,\mathbf{F}^T+\mathbf{n}^T)\right] \\
  & = & \mathbf{F}\,\mathbf{C}_{m}\,\mathbf{F}^T + \mathbf{C}_n\;, \\
  \mathbf{C}_{md} & = &  E\left[\mathbf{m}\,(\mathbf{m}^T\,\mathbf{F}^T+\mathbf{n}^T)\right] = \mathbf{C}_{m}\,\mathbf{F}^T\;.\end{array}$$
"""

# ╔═╡ 27c43328-98f7-4b3b-8e50-3f1ba4bb7ffc
md"""
Putting it all together, we find the following expression for the minimum-variance linear estimate
in the case of linear forward modeling with additive noise:

$$\widehat{\mathbf{m}} = \mathbf{C}_{m}\,\mathbf{F}^T\,\left(\mathbf{F}\,\mathbf{C}_{m}\,\mathbf{F}^T + \mathbf{C}_n\right)^{-1}\,\mathbf{d}\;.$$
"""

# ╔═╡ f655b9d2-db11-4b2c-beb7-a94cae6df289
md"""
Assuming that the covariance matrices $\mathbf{C}_{m}$ and $\mathbf{C}_{n}$ are invertible, there is an alternative form of the last equation. Considering the matrix

$$\mathbf{W} = \mathbf{F}^T\,\mathbf{C}_n^{-1}\,\mathbf{F}\,\mathbf{C}_{m}\,\mathbf{F}^T + \mathbf{F}^T\;,$$ 
we can see that $\mathbf{W}$ could be factored in two different ways, as follows:
$$\begin{array}{rcl}\mathbf{W} & = & \mathbf{F}^T\,\mathbf{C}_n^{-1}\,\left(\mathbf{F}\,\mathbf{C}_{m}\,\mathbf{F}^T + \mathbf{C}_n\right) \\ & = & \left(\mathbf{F}^T\,\mathbf{C}_n^{-1}\,\mathbf{F} + \mathbf{C}_{m}^{-1}\right)\,\mathbf{C}_{m}\,\mathbf{F}^T\;.\end{array}$$
"""

# ╔═╡ 43ac8673-9eb9-4c79-b4aa-68749b68e104
md"""
Therefore,

$$\begin{array}{c} \mathbf{C}_{m}\,\mathbf{F}^T\,\left(\mathbf{F}\,\mathbf{C}_{m}\,\mathbf{F}^T + \mathbf{C}_n\right)^{-1} \\ = \left(\mathbf{F}^T\,\mathbf{C}_n^{-1}\,\mathbf{F} + \mathbf{C}_{m}^{-1}\right)^{-1}\,\mathbf{F}^T\,\mathbf{C}_n^{-1}\;,\end{array}$$

which proves the following alternative, algebraically equivalent, form of the model estimate:

$$\widehat{\mathbf{m}} = \left(\mathbf{F}^T\,\mathbf{C}_n^{-1}\,\mathbf{F} + \mathbf{C}_{m}^{-1}\right)^{-1}\,\mathbf{F}^T\,\mathbf{C}_n^{-1}\,\mathbf{d}\;.$$
"""

# ╔═╡ 2a5eec20-6da2-4111-ae3b-28a44f104a62
md"""
!!! note

    There are two different but algebraically equivalent formulations of minimum-variance linear estimation. One uses the model and noise covariance matrices, and the other uses the corresponding inverse covariance matrices.
"""

# ╔═╡ 1faa96ee-9939-4115-a9f6-3c6aebebc5f1
md"""
## Bias and prior

The specified linear estimates have minimum variance but may have a bias. To guarantee an unbiased estimate (known as BLUE: *Best Linear Unbiased Estimator*), we need to require that $E[\mathbf{m}] = \mathbf{0}$ and $E[\mathbf{n}] = \mathbf{0}$. A zero-mean model is not always an easy requirement to satisfy. The closest we can get in practice is to have a starting model or *prior*, $\mathbf{m}_0$, and to estimate the deviation from the prior. The resultant estimation equation is
"""

# ╔═╡ 048f0441-5706-430b-a50d-ed930178cc6e
md"""

$$\begin{array}{rcl} \widehat{\mathbf{m}} & = & 
\mathbf{m}_0 + \mathbf{C}_{m}\,\mathbf{F}^T\,\left(\mathbf{F}\,\mathbf{C}_{m}\,\mathbf{F}^T + \mathbf{C}_n\right)^{-1}\,(\mathbf{d}-\mathbf{F}\,\mathbf{m}_0)\\
& = & \mathbf{m}_0 + \left(\mathbf{F}^T\,\mathbf{C}_n^{-1}\,\mathbf{F} + \mathbf{C}_{m}^{-1}\right)^{-1}\,\mathbf{F}^T\,\mathbf{C}_n^{-1}\,(\mathbf{d}-\mathbf{F}\,\mathbf{m}_0)\;.\end{array}$$
"""

# ╔═╡ af41dadc-e010-409e-b860-f3983a56eaeb
md"""
## Connection with least squares

The two alternative formulations of minimum-variance inversion are inherently linked to two different versions of least-squares optimization.
"""

# ╔═╡ 28663346-7e65-41de-a803-d04ecfe1a766
md"""
### Overdetermined least squares

If an $m \times n$ "skinny" rectangular matrix $\mathbf{A}$ has a rank $n < m$, it might be impossible to find an $n$-component vector $\mathbf{x}$ that satisfies the linear equation
$\mathbf{A}\,\mathbf{x} = \mathbf{b}$ exactly. In other words, in this linear system, we have more equations than unknowns. The best we can do is to try satisfying an approximate equation $\mathbf{A}\,\mathbf{x} \approx \mathbf{b}$ and to look for its solution, for example, by minimizing the least-square measure of the data misfit $\mathbf{r}=\mathbf{A}\,\mathbf{x}-\mathbf{b}$. 
"""

# ╔═╡ 3e611c58-b6d5-4371-80e7-b9e52c0d91ab
md"""
Algebraic manipulations show that

$$\begin{array}{rcl} \nonumber
|\mathbf{r}|^2 & = & \mathbf{r}^T\,\mathbf{r} = (\mathbf{A}\,\mathbf{x}-\mathbf{b})^T\,(\mathbf{A}\,\mathbf{x}-\mathbf{b}) \\
& = & \mathbf{x}^T\,\mathbf{A}^T\,\mathbf{A}\,\mathbf{x} - \mathbf{x}^T\,\mathbf{A}^T\,\mathbf{b} - \mathbf{b}^T\,\mathbf{A}\,\mathbf{x} + \mathbf{b}^T\,\mathbf{b} \\
\nonumber
& = & \left|\mathbf{A}\,\left[\mathbf{x}-(\mathbf{A}^T\,\mathbf{A})^{-1}\,\mathbf{A}^T\,\mathbf{b}\right]\right|^2 \\ & & - \mathbf{b}^T\,\mathbf{A}\,(\mathbf{A}^T\,\mathbf{A})^{-1}\,\mathbf{A}^T\,\mathbf{b} + \mathbf{b}^T\,\mathbf{b}\;,\end{array}$$

which implies that the least-squares minimum is achieved when

$$\widehat{\mathbf{x}}=(\mathbf{A}^T\,\mathbf{A})^{-1}\,\mathbf{A}^T\,\mathbf{b}\;.$$
"""

# ╔═╡ a8519b23-2753-4356-8c71-82574a1fa759
md"""
We can make the last equation equivalent to the previously derived equation by setting $\mathbf{x}=\mathbf{m}$ and defining $\mathbf{A}$ as a composite block-column matrix
$$\mathbf{A} = \left[\begin{array}{l} \mathbf{D}_n\,\mathbf{F} \\ \mathbf{D}_m \end{array}\right]\;,$$

and $\mathbf{b}$ as a composite block-column vector
$$\mathbf{b} = \left[\begin{array}{l} \mathbf{D}_n\,\mathbf{d} \\ \mathbf{0} \end{array}\right]\;,$$

where $\mathbf{D}_n$ and $\mathbf{D}_m$ are chosen so that
$\mathbf{D}_n^T\,\mathbf{D}_n=\mathbf{C}_n^{-1}$ and
$\mathbf{D}_m^T\,\mathbf{D}_m=\mathbf{C}_m^{-1}$. 
"""

# ╔═╡ cb0998ef-a159-4c46-8c0b-19be4bc60116
md"""
In other words, the
previously derived minimum-variance estimate is equivalent to the
result of minimizing the composite least-squares objective function

$$\begin{array}{rcl}G_1(\mathbf{m}) & = & \left|\mathbf{D}_n\,(\mathbf{F}\,\mathbf{m}-\mathbf{d})\right|^2+\left|\mathbf{D}_m\,\mathbf{m}\right|^2 \\ & = &
(\mathbf{F}\,\mathbf{m}-\mathbf{d})^T\,\mathbf{C}_n^{-1}\,(\mathbf{F}\,\mathbf{m}-\mathbf{d})+\mathbf{m}^T\,\mathbf{C}_m^{-1}\,\mathbf{m}\;.\end{array}$$

The first term relates to the noise-covariance-weighted data misfit. The second term is known as the *model regularization* term. The estimation problem in this setup aims to minimize both terms together by minimizing their sum.
"""

# ╔═╡ ea01ecad-1084-49ce-83cd-1c3ce5c13d83
md"""
* **Tikhonov, A. N., and V. Y. Arsenin, 1977, Solution of ill-posed problems: John Wiley and Sons.**
* **Engl, H., M. Hanke, and A. Neubauer, 1996, Regularization of inverse problems: Kluwer Academic Publishers.**
"""

# ╔═╡ 36e8a306-3218-4eea-a30a-77ecbb06ae6f
md"""
### Linear regression

A classic example of overdetermined least-squares is the problem of linear regression. If the forward model for the given data is a linear superposition of basis functions

$$d(\mathbf{x}) = \displaystyle \sum_{k=1}^N c_k\,\phi_k(\mathbf{x})\;,$$

then the data $d(\mathbf{x})$ corresponds to the vector $\mathbf{b}$, the unknown coefficients $c_k$ to the model vector $\mathbf{x}$, and the basis set of functions $\phi_k(\mathbf{x})$ to the matrix $\mathbf{A}$, with the optimal least-squares estimate $\widehat{\mathbf{x}}$ given by the equation derived above.
"""

# ╔═╡ 1982fd05-5c9a-40ad-9772-501a4df4fe88
md"""
Let us apply this approach to modeling the summer temperature trends in Austin, as measured at the Camp Mabry weather station.
"""

# ╔═╡ b09841fd-b3eb-4898-9832-f4a96fc465d1
data = DataFrame(CSV.File("mabry.csv"))

# ╔═╡ 136f8680-c2b8-47e1-99f4-378e0f0d14a0
begin
	summer = DataFrame(Year = Int64[], Temperature = Float64[])
	for y in 2015:2025
	    # summer is from June to August
	    df = data[(year.(data.Date) .== y) .& 
	              (month.(data.Date) .>= 6) .& 
	              (month.(data.Date) .<= 8), :]
	    t = median(df.Temperature)
	    push!(summer,[y, t])
	end
	summer
end

# ╔═╡ 8e8f9268-d6a3-4ec9-ac50-7fe33c0c66e4
scatter(summer.Year, summer.Temperature, xticks=summer.Year,
        title="Median Summer Temperature at Camp Mabry",
        ylabel="Degrees Celsius", label=:none)

# ╔═╡ 3820002d-4230-4203-bbe9-077c18e74f07
md"""
Our first model for the data will be a linear function

$T(y) = c_1 + c_2\,y\;,$

where $y$ is year, and $T$ is temperature. The corresponding basis functions are $\phi_1(y)=1$ and $\phi_2(y)=y$.
"""

# ╔═╡ bf5b4c79-1c20-49df-8e0f-a65061ac90e0
A = hcat(ones(11), summer.Year)

# ╔═╡ 5dc11510-28db-4fa6-91ac-fe8d7daa8a26
cs = (A' * A) \ (A'*summer.Temperature)

# ╔═╡ 11c62aa9-6cbb-403e-980a-1b4d0a9d752a
begin
	scatter(summer.Year, summer.Temperature, xticks=summer.Year,
	        title="Median Summer Temperature at Camp Mabry",
	        ylabel="Degrees Celsius", xlabel="Year", label=:none)
	plot!(summer.Year, A*cs, label="least-squares fit")
end

# ╔═╡ afa92f44-9c0b-4475-861e-415c277fb76f
F(c1, c2) = sum((c1 .+ c2*summer.Year - summer.Temperature) .^ 2)

# ╔═╡ 525a44c1-58f4-4187-82cd-2ed343c03fc7
md"""
!!! assignment
    ## Task 2

    The least-squares objective function as a function of $c_1$ and $c_2$ is defined as

    $$F(c_1,c_2) = \displaystyle \sum\limits_n \left(c_1 + c_2\,y_n - t_n\right)^2\;.$$

    Plot $F(c_1, 0)$ over the range of $c_1$ from $-1000$ to $1000$. Plot $F(0, c_2)$ over the range of $c_2$ from $-10$ to $10$. Finally, create a two-dimensional plot of $F(c_1, c_2)$ using either `heatmap` or `contour`: [https://docs.juliaplots.org/latest/series_types/contour/](https://docs.juliaplots.org/latest/series_types/contour/)
"""

# ╔═╡ 9fa35f15-e42f-4c05-9eeb-1acd4dc9777a
md"""
We can extend the model by using a parabola

$T(y) = c_1 + c_2\,y + c_3\,y^2\;.$

The corresponding basis functions are $\phi_1(y)=1$,  $\phi_2(y)=y$, and $\phi_3(y)=y^2$.
"""

# ╔═╡ 8cb8c661-d542-40cf-96a5-2725c1b0386e
A2 = hcat(ones(11), summer.Year, summer.Year .^ 2)

# ╔═╡ 90601c53-8fc6-49d1-8b10-a26e2f9fc45d
cs2 = (A2' * A2) \ (A2'*summer.Temperature)

# ╔═╡ 30fc1f7e-6b9c-4271-9fdf-2fdeddeaa3b9
begin
	scatter(summer.Year, summer.Temperature, 
	        title="Median Summer Temperature at Camp Mabry",
	        ylabel="Degrees Celsius", xlabel="Year", label=:none)
	plot!(summer.Year, A2*cs2, xticks=summer.Year, label="least-squares fit")
end

# ╔═╡ 49b0a46f-d50d-43c9-a30c-014984414a8c
md"""
### Data example: Global mean sea level

*Since 1993, measurements from the TOPEX/Poseidon and Jason series (Jason-1, Jason-2, and Jason-3) of satellite radar altimeter missions have allowed estimates of global mean sea level (e.g., sea surface height averaged globally over the planet).<...>
Global mean sea level is an excellent barometer of climate change because many of the natural variations of sea level average out in the global mean. <...> In combination with satellite gravity data (GRACE), one can parse out the major contributions to the observed 3.3 mm/year increase in global mean sea level (thermal expansion, Greenland, Antarctica, mountain glaciers).*
[https://climatedataguide.ucar.edu/climate-data/global-mean-sea-level-topex-jason-altimetry](https://climatedataguide.ucar.edu/climate-data/global-mean-sea-level-topex-jason-altimetry)
"""

# ╔═╡ bf5d528c-ed4c-4feb-989b-a7d59b13db83
md"""
![](https://sealevel.nasa.gov/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBdG9JIiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--3e1a236908e62f930b6f68357cf95ce5518f18ef/jason3490.jpeg?disposition=inline)

[https://www.jpl.nasa.gov/missions/jason-3](https://www.jpl.nasa.gov/missions/jason-3)
"""

# ╔═╡ af119861-7797-4bd2-8c85-8fead45f8783
begin
	url = "https://sealevel.colorado.edu/files/2025_rel1/gmsl_2025rel1_seasons_retained.txt"
	sea = readdlm(HTTP.get(url).body, skipstart=1)
end

# ╔═╡ 8904014a-ac9c-40c5-afcf-09cccc1d71ec
GMSL = DataFrame(sea, ["Year", "Level"])

# ╔═╡ 4e72b5fc-c6c1-403b-bdba-08a664fbd5ae
plot(GMSL.Year, GMSL.Level, 
     title="Global Mean Sea Level", linewidth=2,
     ylabel="Level (mm)", xlabel="Year", label=:none)

# ╔═╡ 305518f8-0b0e-4996-9a04-b2c99763f9e5
md"""
!!! assignment
    ## Task 3

    Use linear regression to fit a linear trend to the GMSL data and visualize the result.
"""

# ╔═╡ 50764a2a-19e4-46cd-a8a4-8e7b9418d98f
md"""
!!! assignment
    ## Task 4

    Use linear regression to fit a quadratic trend to the GMSL data and visualize the result.
"""

# ╔═╡ b8d9d27d-3f95-49a2-89b3-855cafc69790
md"""
!!! assignment
    ## Task 5

    Along with the linear or quadratic trend showing the overall rise in sea level, we can see oscillations caused by seasonal changes. Expand the regression model to include these variations.

    To do that, in addition to $\phi_1(y)=1$,  $\phi_2(y)=y$, and $\phi_3(y)=y^2$, consider basis functions of the form $\sin(2\pi\,c\,y)$ and $\cos(2\pi\,c\,y)$, where $c$ corresponds to the number of cycles per year. Add basis functions for $c$ in the range from 1 to 6 and test whether they improve the model.
"""

# ╔═╡ 1e07372c-5456-4acb-b2e4-b0312287b55b
plot(GMSL.Year, sin.(2π*2*GMSL.Year), 
     xlabel="Year", label="two cycles per year")

# ╔═╡ 2ad5772d-e940-4e0d-90c4-1646ae714e9a
md"""
!!! assignment
    ## Bonus Task

    Along with the rising trend and seasonal changes, the global sea level is also influenced by ENSO (El Niño-Southern Oscillation), a recurring climate pattern with warm and cool phases (El Niño and La Niña). For the extra task, extract ENSO data and add it as an extra basis function in the regression model.

    **Wilson, C.R., 2021. Essentials of geophysical data processing. Cambridge University Press.**
"""

# ╔═╡ 1c43ed22-6ace-4702-b9d1-305be882a031
begin
	enso_url = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"
	# exctract tables from a web page
	tables = scrape_tables(enso_url)
end

# ╔═╡ 7a7fb59a-9b04-43a7-8c85-b0183d37993e
table = tables[end-2];

# ╔═╡ 8a71e1fc-4164-4948-9656-a5682cf77509
begin
	ENSO = DataFrame([name => Float32[] for name in table.rows[1]]);
	begin
		for row in table.rows
		    if row[1] != "Year" && length(row) == 13
		        push!(ENSO, map(x -> parse(Float32, x), row))
		    end
		end
		ENSO.Year = Int32.(ENSO.Year)
		ENSO
	end
end

# ╔═╡ d5d7801f-592c-4c08-82e7-9aefc06ba7c2
enso = vec(Matrix(ENSO[:,2:end])');

# ╔═╡ 8c71f695-fa6a-4f68-a4dd-d4392adafa4f
years = range(start=1950+1/24, step=1/12, length=length(enso));

# ╔═╡ fee68535-5090-4f35-8f45-ab8cf5249e4e
plot(years[500:end], enso[500:end], linewidth=2, line_z=enso[500:end],
     xlabel="Year", ylabel="Temperature Anomaly (Degrees)", 
     label=:none, title="ENSO", color=:coolwarm, ylim=(-2.6,2.6))

# ╔═╡ 9ef07d97-b596-4bbb-a269-e5de2d514ed8
md"""
To complete the bonus task, interpolate the ENSO data to match the sampling used in the GMSL data and incorporate it as an additional basis function in the linear regression.
"""

# ╔═╡ c625e9f2-3750-44cb-aba6-8847e4cba375
md"""
### Underdetermined least squares
    
If an $m \times n$ rectangular matrix $\mathbf{A}$ has a rank $m < n$, there might be infinitely many $n$-component vectors $\mathbf{x}$ that satisfy the linear equation $\mathbf{A}\,\mathbf{x} = \mathbf{b}$ exactly. In other words, in this linear system, we have fewer equations than unknowns. Let us try to represent a solution of the system as $\mathbf{x} = \mathbf{A}^T\,\mathbf{y}$, where $\mathbf{y}$ is another unknown vector. 
"""

# ╔═╡ b40fb6e3-52a3-452f-adfb-b95a18ffaec0
md"""
Substituting this trial solution, we find that

$$\mathbf{A}\,\mathbf{A}^T\,\mathbf{y} = \mathbf{b}$$

and, therefore,

$$\widehat{\mathbf{x}}
= \mathbf{A}^T\,\mathbf{y} = \mathbf{A}^T\,\left(\mathbf{A}\,\mathbf{A}^T\right)^{-1}\,\mathbf{b}\;.$$

This solution satisfies the equation but is by no means unique. 
"""

# ╔═╡ 1c8b33d4-bdfd-4ae7-8dd6-e537051976f2
md"""
Let us consider another solution $\mathbf{x}=\widehat{\mathbf{x}}+\delta\mathbf{x}$ such that $\mathbf{A}\,\delta\mathbf{x} = \mathbf{0}$ and look at its length

$$|\widehat{\mathbf{x}}+\delta\mathbf{x}|^2 = |\widehat{\mathbf{x}}|^2+|\delta\mathbf{x}|^2 + 2\,\widehat{\mathbf{x}}^T\,\delta\mathbf{x}\;.$$

We can notice that 

$$2\,\widehat{\mathbf{x}}^T\,\delta\mathbf{x} = \mathbf{b}^T\,\left(\mathbf{A}\,\mathbf{A}^T\right)^{-1}\,\mathbf{A}\,\delta\mathbf{x}=\mathbf{0}\;.$$

Therefore, $|\widehat{\mathbf{x}}+\delta\mathbf{x}|^2 =|\widehat{\mathbf{x}}|^2+|\delta\mathbf{x}|^2 \ge |\widehat{\mathbf{x}}|^2$. In other words, $\widehat{\mathbf{x}}$ defines a solution that has, among the infinitely many solutions of the underdetermined system, the minimum length.
"""

# ╔═╡ 6cf53e76-449c-4958-a422-b1a43cceb8ad
md"""
We can make equation the new estimation equation equivalent to the previously derived by using the following definitions: $\mathbf{b}=\mathbf{d}$, $\mathbf{A}$ is a composite block-row matrix

$$\mathbf{A} = \left[\begin{array}{ll} \mathbf{F}\,\mathbf{P}_m & \mathbf{P}_n\end{array}\right]\;,$$

$\mathbf{x}$ is a composite block-column vector: 

$$\mathbf{x} = \left[\begin{array}{l} \mathbf{x}_p \\ \mathbf{x}_n \end{array}\right]\;,$$

$$\mathbf{m} = \mathbf{P}_m\,\mathbf{x}_p\;,$$

$$\mathbf{n} = \mathbf{P}_m\,\mathbf{x}_n\;,$$ 

and $\mathbf{P}_m$ and $\mathbf{P}_n$ are chosen so that $\mathbf{P}_n\,\mathbf{P}_n^T=\mathbf{C}_n$ and $\mathbf{P}_m\,\mathbf{P}_m^T=\mathbf{C}_m$. 
"""

# ╔═╡ 7feb89df-452d-486f-8eb1-a01c6b26e955
md"""
In other words, the minimum-variance estimate is equivalent to the result of minimizing the length of the composite vector

$$G_2(\mathbf{x}) = |\mathbf{x}_p|^2+|\mathbf{x}_n|^2$$

under the linear constraint

$$\mathbf{F}\,\mathbf{P}_m\,\mathbf{x}_p + \mathbf{P}_n\,\mathbf{x}_n = \mathbf{d}\;,$$

followed by the evaluation $\widehat{\mathbf{m}} = \mathbf{P}_m\,\widehat{\mathbf{x}_p}$. We have found an alternative formulation of the linear estimation problem using under-determined least squares.
"""

# ╔═╡ 15c5c522-6cb0-4d4c-9460-946887ffcb66
md"""
## Signal and noise separation

Suppose that our data consists of statically-independent signal and noise

$$\mathbf{d = s + n}\;,$$

and we happen to know the signal and noise covariances $\mathbf{C}_s$ and $\mathbf{C}_n$. In this formulation, the signal-and-noise separation problem corresponds to the case of the forward operator $\mathbf{F}$ being simply an identity. 
"""

# ╔═╡ d4672a29-4457-44b6-b3a0-4b0aeb609487
md"""
The solution is 

$$\begin{array}{rcl} \widehat{\mathbf{s}} & = & \mathbf{C}_{s}\,\left(\mathbf{C}_{s} + \mathbf{C}_n\right)^{-1}\,\mathbf{d} \\
  \widehat{\mathbf{n}} & = & \mathbf{C}_{n}\,\left(\mathbf{C}_{s} + \mathbf{C}_n\right)^{-1}\,\mathbf{d}\;,\end{array}$$

which follows from the first formulation of linear estimation. Alternatively, using the inverse covariance operators, we can also write the solution as

$$\begin{array}{rcl} \widehat{\mathbf{s}} & = & \left(\mathbf{C}_n^{-1} + \mathbf{C}_{s}^{-1}\right)^{-1}\,\mathbf{C}_n^{-1}\,\mathbf{d} \\
  \widehat{\mathbf{n}} & = & \left(\mathbf{C}_n^{-1} + \mathbf{C}_{s}^{-1}\right)^{-1}\,\mathbf{C}_s^{-1}\,\mathbf{d}\;.\end{array}$$
"""

# ╔═╡ ed230dc0-e0e8-4d35-be5e-c4ea60685665
md"""
* **Wiener, N., 1942, The interpolation, extrapolation and smoothing of stationary time series: Technical report, NDRC Report, Cambridge.**
"""

# ╔═╡ 5b52085f-9729-4cf4-9782-c7ed58fdd4e0
md"""
![](https://static01.nyt.com/images/2013/05/21/science/21WEIN1_SPAN/21WEIN1-superJumbo.jpg?quality=75&auto=webp)

[https://en.wikipedia.org/wiki/Norbert_Wiener](https://en.wikipedia.org/wiki/Norbert_Wiener)
"""

# ╔═╡ d8e709ae-4100-4775-a684-f1889de9cb75
md"""
## Gaussian probabilities

Previously, when examining the problem of calculating the mean value, we established a close link between least-squares estimation and Gaussian probability density functions. The broader case of linear estimation further expands this connection. Assume that, without any data, we can model our system using the Gaussian probability density function

$$P(\mathbf{m}) = \frac{1}{\sqrt{(2\pi)^N\,|\mathbf{C}_m|}}\,\exp\left[-\mathbf{m}^T\,\mathbf{C}_m^{-1}\,\mathbf{m}\right]\;.$$

For simplicity, we can assume that the bias has been subtracted from the model so that $\mathbf{m}$ now has a zero expectation.
"""

# ╔═╡ 59a2cf18-13d5-47b7-b0ec-45c7af0d57ed
md"""
Next, the data are acquired with the probability density

$$P(\mathbf{d}) = \frac{1}{\sqrt{(2\pi)^N\,|\mathbf{C}_n|}}\,\exp\left[-(\mathbf{F}\,\mathbf{m}-\mathbf{d})^T\,\mathbf{C}_{n}^{-1}\,(\mathbf{F}\,\mathbf{m}-\mathbf{d})\right]\;.$$
"""

# ╔═╡ 9a433464-a590-41d1-bae9-d4b11e1eca34
md"""
The *posterior* probability of $\mathbf{m}$ given $\mathbf{d}$, according to the Bayes theorem, is the product of the two probabilities

$$\begin{array}{rcl}
\nonumber
P(\mathbf{m}|\mathbf{d}) & = & \displaystyle \frac{1}{(2\pi)^N\,\sqrt{|\mathbf{C}_m|\,|\mathbf{C}_n|}}\,\exp\left[-\mathbf{m}^T\,\mathbf{C}_m^{-1}\,\mathbf{m} \right. 
\\ & &
\displaystyle \left. -(\mathbf{F}\,\mathbf{m}-\mathbf{d})^T\,\mathbf{C}_{n}^{-1}\,(\mathbf{F}\,\mathbf{m}-\mathbf{d})\right] \\
& = & \hat{A}\,\exp\left[-(\mathbf{m}-\widehat{\mathbf{m}})^T\,\widehat{\mathbf{C}_m}^{-1}\,(\mathbf{m}-\widehat{\mathbf{m}})\right]\;,\end{array}$$

where $\hat{A}$ absorbs scaling factors that do not depend on $\mathbf{m}$. 
"""

# ╔═╡ 8e7a0674-5c91-4077-9982-9fd9c441bae8
md"""
We can see that, under the assumption of normal distributions, the new model remains normally distributed.  If the prior expectation of $\mathbf{m}$ was zero, the new expectation of $\mathbf{m}$ is given by $\widehat{\mathbf{m}}$. If the prior model covariance was $\mathbf{C}_m$, the posterior model covariance is

$$\begin{array}{rcl}\widehat{\mathbf{C}_m} & = & \left(\mathbf{F}^T\,\mathbf{C}_n^{-1}\,\mathbf{F} + \mathbf{C}_{m}^{-1}\right)^{-1} \\ & = &
\mathbf{C}_m-\mathbf{C}_m\,\mathbf{F}^T\,\left(\mathbf{F}\,\mathbf{C}_{m}\,\mathbf{F}^T + \mathbf{C}_n\right)^{-1}\,\mathbf{F}\,\mathbf{C}_m\;.\end{array}$$

According to the Gauss-Markov minimum-variance principle, the covariance decreases after linear estimation. From the perspective of Bayesian statistics, this reduction occurs because the data provides new information about the model.
"""

# ╔═╡ 68337c04-2449-4c5b-b592-23d613613f8f
md"""
Those who adopt the Bayesian approach to inversion argue that a solution to an estimation problem should be described not just as a single expected value (the most probable answer), but as the complete probability distribution of the estimate, including its inherent uncertainty.

* **Tarantola, A., 2005, Inverse Problem Theory and methods for model parameter estimation: SIAM.**
* **Wunsch, C., 2006, Discrete inverse and state estimation problems: Cambridge University Press.**
"""

# ╔═╡ 2110837d-be01-4302-a059-7ad10fa929b0
md"""
## Road ahead

Defining an optimal linear estimate as the one with the lowest variance, we have shown that this estimate corresponds to the least-squares solution of either an overdetermined or underdetermined linear system. It also corresponds to maximizing the Bayesian probability under the assumption of Gaussian distributions.
"""

# ╔═╡ 9a259eb4-97a2-4f0d-a2e7-fd41e3f8d2df
md"""
Where do we go from here? For an $N \times N$ dense matrix, matrix multiplication requires $O(N^2)$ numerical operations. In contrast, matrix inversion demands $O(N^3)$ operations. As previously discussed, this cost is unaffordable in large-scale practical problems. Our next goal is to find meaningful linear operators that can be executed in $O(N)$ or, at most, $O(N \log N)$ operations. Instead of performing an exact matrix inversion, we will also explore iterative inversion algorithms that need only $O(N)$ operations per iteration and can produce a reasonable result after just a few iterations. When considering iterative inversion, it will become clear that although algebraically equivalent, different formulations may exhibit different numerical behavior in practice.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
GMT = "5752ebe1-31b9-557e-87aa-f909b540aa54"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUIExtra = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
TableScraper = "3d876f86-fca9-45cb-9864-7207416dc431"

[compat]
CSV = "~0.10.15"
DataFrames = "~1.7.1"
GMT = "~1.22.4"
HTTP = "~1.10.19"
Plots = "~1.40.20"
PlutoUIExtra = "~0.1.8"
TableScraper = "~0.1.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "25b02fbf2300ff6c65e25125bb347af9c12987f8"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

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

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.Cascadia]]
deps = ["AbstractTrees", "Gumbo"]
git-tree-sha1 = "c0769cbd930aea932c0912c4d2749c619a263fc1"
uuid = "54eefc05-d75b-58de-a785-1a3403f0919f"
version = "1.0.2"

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
git-tree-sha1 = "a37ac0840a1196cd00317b57e39d6586bf0fd6f6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.1"

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

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

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
git-tree-sha1 = "1bb6366020b59ed182775a88905bd4e59699c7af"
uuid = "5752ebe1-31b9-557e-87aa-f909b540aa54"
version = "1.22.4"
weakdeps = ["DataFrames"]

    [deps.GMT.extensions]
    GMTDataFramesExt = "DataFrames"

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

[[deps.Gumbo]]
deps = ["AbstractTrees", "Gumbo_jll", "Libdl"]
git-tree-sha1 = "eab9e02310eb2c3e618343c859a12b51e7577f5e"
uuid = "708ec375-b3d6-5a57-a7ce-8257bf98657a"
version = "0.8.3"

[[deps.Gumbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "29070dee9df18d9565276d68a596854b1764aa38"
uuid = "528830af-5a63-567c-a44a-034ed33b8444"
version = "0.10.2+0"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "82a471768b513dc39e471540fdadc84ff80ff997"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.3+3"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e6fe50ae7f23d171f44e311c2960294aaa0beb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.19"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

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

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "fa7fd067dca76cadd880f1ca937b4f387975a9f5"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.16.0+0"

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

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "9341048b9f723f2ae2a72a5269ac2f15f80534dc"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.3.2+0"

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

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "libpng_jll"]
git-tree-sha1 = "215a6666fee6d6b3a6e75f2cc22cb767e2dd393a"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.5.5+0"

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
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "bfe839e9668f0c58367fb62d8757315c0eac8777"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.20"

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
git-tree-sha1 = "3faff84e6f97a7f18e0dd24373daa229fd358db5"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.73"

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

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

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
git-tree-sha1 = "a136f98cefaf3e2924a66bd75173d1c891ab7453"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.7"

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

[[deps.TableScraper]]
deps = ["Cascadia", "Gumbo", "HTTP", "Tables"]
git-tree-sha1 = "73e600bad3a9b6c04c8a055e316fd60dd2ab372c"
uuid = "3d876f86-fca9-45cb-9864-7207416dc431"
version = "0.1.4"

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

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "6258d453843c466d84c17a58732dda5deeb8d3af"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.24.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "af305cc62419f9bd61b6644d19170a4d258c7967"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.7.0"

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
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

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
# ╟─e5d0c763-9d85-4154-8a4a-df0c95d54d28
# ╟─9c2b8cc5-0f43-4d12-8261-24cff9361204
# ╟─3252a716-33bd-42e9-a3ec-8ccb4111de24
# ╟─235cea6e-05a8-4132-bf79-ffb3c82a550f
# ╟─679977f5-7e27-481b-8584-55b711539e08
# ╟─a3a7f340-c8ff-4d1a-89d5-90a42d627994
# ╟─723a5475-649b-4e8b-9471-027ad86fad65
# ╟─8777da1d-d9cc-448c-924d-5c87e4b531b0
# ╟─be14b720-b4eb-4a32-b5ec-a2e404a5cc92
# ╟─cec6d5ba-638f-4047-ba11-aad35b55774c
# ╟─48fc15f0-9aa2-432a-9ddc-f6b9877b7df5
# ╠═bbe25e38-95f1-4ed9-8415-d92ca62592a3
# ╟─40394b16-670b-42c8-8c84-7db80a60a7da
# ╠═c2579001-ce17-419a-84f5-f2e98db93e8b
# ╟─77773c15-4e89-4d96-9911-39703a51efd6
# ╟─acbb8073-2fc0-411f-84cb-be9523844829
# ╠═35b24d3b-1c01-44d4-b681-8768ce054240
# ╠═cd323d29-43ca-4ce7-aa4b-1110bbfc982d
# ╠═fc18f856-5fd6-4f43-9a21-f67ff71a07e2
# ╠═16a4a86e-4cc8-4f2d-9a3d-76d58fdc4fa2
# ╠═590e20ad-283c-4114-a4fe-15c8dbe76c8e
# ╠═356dd92d-bf99-48aa-b78f-ea33c1bf0e74
# ╠═f5ea975d-c191-4159-aebe-1f900238ac43
# ╠═6a581b1f-8ed8-4713-b071-9ef13cfd811d
# ╠═c4df9fab-37f0-42d4-b96c-f029995b516b
# ╠═d1cefa45-272a-40cc-9b8c-f070e7cb1e71
# ╠═12c0a2c3-1938-4ddc-b9f3-86950e3bf1c5
# ╟─7b82027b-db18-4d3d-bcd4-9ee4c595d945
# ╟─35749e12-e4e1-4f5f-b949-b2f71cf6ebd2
# ╟─f3aa8430-2664-4ec5-9844-0c7e0de84321
# ╟─bd90f64c-7ecf-45e5-9d37-36d90703e58e
# ╟─74bf6172-aae1-40fa-a00a-f99e4d3014a6
# ╠═dd34aa6d-5673-41ac-9e6f-ac60a3cf0978
# ╠═aabe9f28-72bc-4d49-8884-b3831371cd51
# ╠═4be29b22-b5d4-4dad-977d-6c3d581011db
# ╠═c356de4c-1c4f-4961-95d0-a56bf7f82404
# ╠═1fc47b85-cd2f-42d0-af66-c0440a2ded00
# ╠═402cd0a6-ec38-48d3-8177-b73975a48ab2
# ╠═9776b151-6f85-404c-a19b-b4312748b6fc
# ╠═13a8c867-7dd7-4f12-b63a-63e871dcbf40
# ╠═14f59e33-fd48-4711-a1c4-674610781080
# ╠═e0ef5c6e-489b-4ee8-afd1-6341df9b8911
# ╠═8e77a9f1-ebee-4578-b4d4-4b8c268a8259
# ╠═616684c7-d082-4d03-9f2d-3c5d3f8ec1d4
# ╟─26a14a97-62fe-41cf-9e95-36f509dd4b58
# ╟─3d81afc6-f055-4262-84b9-49baf846339c
# ╟─6176ef7f-17fc-4f93-bcbf-2cd230f77b1a
# ╟─27c43328-98f7-4b3b-8e50-3f1ba4bb7ffc
# ╟─f655b9d2-db11-4b2c-beb7-a94cae6df289
# ╟─43ac8673-9eb9-4c79-b4aa-68749b68e104
# ╟─2a5eec20-6da2-4111-ae3b-28a44f104a62
# ╟─1faa96ee-9939-4115-a9f6-3c6aebebc5f1
# ╟─048f0441-5706-430b-a50d-ed930178cc6e
# ╟─af41dadc-e010-409e-b860-f3983a56eaeb
# ╟─28663346-7e65-41de-a803-d04ecfe1a766
# ╟─3e611c58-b6d5-4371-80e7-b9e52c0d91ab
# ╟─a8519b23-2753-4356-8c71-82574a1fa759
# ╟─cb0998ef-a159-4c46-8c0b-19be4bc60116
# ╟─ea01ecad-1084-49ce-83cd-1c3ce5c13d83
# ╟─36e8a306-3218-4eea-a30a-77ecbb06ae6f
# ╟─1982fd05-5c9a-40ad-9772-501a4df4fe88
# ╠═a42162f8-4c1f-492f-98e4-1590b1a915bc
# ╠═b09841fd-b3eb-4898-9832-f4a96fc465d1
# ╠═39e2a830-fd2b-4ebf-81f4-f65cf4556fa8
# ╠═136f8680-c2b8-47e1-99f4-378e0f0d14a0
# ╠═8e8f9268-d6a3-4ec9-ac50-7fe33c0c66e4
# ╟─3820002d-4230-4203-bbe9-077c18e74f07
# ╠═bf5b4c79-1c20-49df-8e0f-a65061ac90e0
# ╠═5dc11510-28db-4fa6-91ac-fe8d7daa8a26
# ╠═11c62aa9-6cbb-403e-980a-1b4d0a9d752a
# ╠═afa92f44-9c0b-4475-861e-415c277fb76f
# ╟─525a44c1-58f4-4187-82cd-2ed343c03fc7
# ╟─9fa35f15-e42f-4c05-9eeb-1acd4dc9777a
# ╠═8cb8c661-d542-40cf-96a5-2725c1b0386e
# ╠═90601c53-8fc6-49d1-8b10-a26e2f9fc45d
# ╠═30fc1f7e-6b9c-4271-9fdf-2fdeddeaa3b9
# ╟─49b0a46f-d50d-43c9-a30c-014984414a8c
# ╟─bf5d528c-ed4c-4feb-989b-a7d59b13db83
# ╠═233e8fad-edf2-45bc-9a7a-6de5bafc902b
# ╠═af119861-7797-4bd2-8c85-8fead45f8783
# ╠═8904014a-ac9c-40c5-afcf-09cccc1d71ec
# ╠═4e72b5fc-c6c1-403b-bdba-08a664fbd5ae
# ╟─305518f8-0b0e-4996-9a04-b2c99763f9e5
# ╟─50764a2a-19e4-46cd-a8a4-8e7b9418d98f
# ╟─b8d9d27d-3f95-49a2-89b3-855cafc69790
# ╠═1e07372c-5456-4acb-b2e4-b0312287b55b
# ╟─2ad5772d-e940-4e0d-90c4-1646ae714e9a
# ╠═e63ea941-0715-4301-89bc-a6d9bed91065
# ╠═1c43ed22-6ace-4702-b9d1-305be882a031
# ╠═7a7fb59a-9b04-43a7-8c85-b0183d37993e
# ╠═8a71e1fc-4164-4948-9656-a5682cf77509
# ╠═d5d7801f-592c-4c08-82e7-9aefc06ba7c2
# ╠═8c71f695-fa6a-4f68-a4dd-d4392adafa4f
# ╠═fee68535-5090-4f35-8f45-ab8cf5249e4e
# ╟─9ef07d97-b596-4bbb-a269-e5de2d514ed8
# ╟─c625e9f2-3750-44cb-aba6-8847e4cba375
# ╟─b40fb6e3-52a3-452f-adfb-b95a18ffaec0
# ╟─1c8b33d4-bdfd-4ae7-8dd6-e537051976f2
# ╟─6cf53e76-449c-4958-a422-b1a43cceb8ad
# ╟─7feb89df-452d-486f-8eb1-a01c6b26e955
# ╟─15c5c522-6cb0-4d4c-9460-946887ffcb66
# ╟─d4672a29-4457-44b6-b3a0-4b0aeb609487
# ╟─ed230dc0-e0e8-4d35-be5e-c4ea60685665
# ╟─5b52085f-9729-4cf4-9782-c7ed58fdd4e0
# ╟─d8e709ae-4100-4775-a684-f1889de9cb75
# ╟─59a2cf18-13d5-47b7-b0ec-45c7af0d57ed
# ╟─9a433464-a590-41d1-bae9-d4b11e1eca34
# ╟─8e7a0674-5c91-4077-9982-9fd9c441bae8
# ╟─68337c04-2449-4c5b-b592-23d613613f8f
# ╟─2110837d-be01-4302-a059-7ad10fa929b0
# ╟─9a259eb4-97a2-4f0d-a2e7-fd41e3f8d2df
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
