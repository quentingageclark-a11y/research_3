### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ 95ac1be7-0da0-40f9-a2dd-dd01dd3f4867
begin
	using PlutoUIExtra
	TableOfContents()
end

# ╔═╡ 7e07cef4-7234-4532-bbc7-f7c36438fd52
using Plots, LaTeXStrings

# ╔═╡ f2937b56-1b31-4710-9433-fc67a4326f93
using BandedMatrices

# ╔═╡ 4720a77f-b7c1-4e6e-ab5f-fbc133b9a4e2
using DelimitedFiles

# ╔═╡ 31279552-8984-4017-affa-404f26064e32
using Random, LinearAlgebra

# ╔═╡ 6f8f1767-8aea-4719-8428-61a973e2417e
import HTTP

# ╔═╡ 753d25ab-f93a-4f20-9a5d-ec83c798f550
let
    notebook_dir = @__DIR__
    toc_path = joinpath(notebook_dir, "TOC_Notebook.jl")
    
    Markdown.parse("[⬅️ Back to Table of Contents](./open?path=$(HTTP.escapeuri(toc_path)))")
end

# ╔═╡ ed4b2c7b-e54b-4dfb-8c69-18e7a2f94311
md"""
# Inverse Interpolation

We previously discussed the problem of interpolating irregularly sampled values from regularly sampled data. Now, using tools and frameworks for linear estimation, we can address the more difficult task of interpolating at regularly sampled locations from irregularly sampled data. This problem is one of the simplest yet practically important examples of inverse problems in data analysis.
"""

# ╔═╡ 9142114d-aab9-4d2c-aebf-c78d74ad60c6
md"""
## Inverse interpolation in 1-D

Let us consider again the toy interpolation example.
"""

# ╔═╡ ead28bad-dfb6-4b43-b144-fe3696133b16
coord = [0.5, 2.0, 2.7, 3.1]

# ╔═╡ 510b8776-83bd-4258-aa4a-02f7d0f2e74c
md"""
In the case of nearest-neighbor interpolation, the connection between
regularly sampled and irregular values is given by

$$\left[\begin{array}{l} y_1 \\ y_2 \\ y_3 \\ y_4\end{array}\right] =
\left[\begin{array}{ccccc} 
1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 & 0
\end{array}\right]\,
\left[\begin{array}{l} x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5\end{array}\right]\;,$$

which can be abbreviated to $\mathbf{y = F\,x}$.
"""

# ╔═╡ 637b2fe3-bc7b-4a39-a122-60153938b6ed
md"""
Reconstructing $\mathbf{x}$ from $\mathbf{y}$ is an inverse problem. If we treat it as a least-squares optimization problem, it will require inverting the matrix

$$\mathbf{F}^T\,\mathbf{F} =
\left[\begin{array}{ccccc} 
1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 2 & 0 \\
0 & 0 & 0 & 0 & 0
        \end{array}\right]\;,$$

which happens to be diagonal with elements along the diagonal that correspond to the *bin count*: how many irregular points are inside each bin in the regularly spaced grid. 
"""

# ╔═╡ f507331e-6bf6-4696-a52b-fc53f0390e05
md"""
Zero values along the diagonal indicate empty bins. In these areas, we lack information about the reconstructed data solely from the input. This information must come from other sources to enable successful reconstruction.
"""

# ╔═╡ 19005a1a-0aa3-43b2-b48f-4e5d3b8cd48a
md"""
This simple example illustrates the core of the inverse interpolation problem: when multiple input values fall within a single output bin, we need to effectively average the information; when there are no values inside an output bin, we have to incorporate additional information (model constraints) from elsewhere. Without this extra information, the most straightforward form of inverse interpolation is *binning*: assigning each bin the average of the data points contained within it.
"""

# ╔═╡ ee067fde-a109-4080-92bc-9f1bf849af02
x = [1, 2, 3, 2, 1]

# ╔═╡ ba61e926-b497-4c4f-a533-09d1552041ae
md"""
!!! assignment
    ## Task 1

    Write a binning program for two-dimensional data by completing the missing lines below.
"""

# ╔═╡ cfea9f3e-097a-4778-8b83-242b817cb2b5
function binning(irreg::Vector{T}, coord, n::Vector{Int}; 
                 d=[1,1], o=[0,0]) where T <: Real
    nd = size(coord, 2)
    reg = zeros(T, n[1], n[2])
    count = zeros(Int, n[1], n[2])
    for id in 1:nd
        x1 = 1 + (coord[1,id] - o[1])/d[1]
        x2 = 1 + (coord[2,id] - o[2])/d[2]
        ## !!! ADD MISSING LINES
    end
    for i in 1:n[1], j in 1:n[2]
        ## !!! ADD MISSING LINES
    end
    return reg
end

# ╔═╡ 429714dc-576b-47a2-87f7-f5f99caa7379
md"""
If we switch to the case of linear interpolation for the forward 
problem, the interpolation matrix takes the form

$$\mathbf{F} =
\left[\begin{array}{ccccc} 
0.5 & 0.5 & 0   & 0   & 0 \\
0   & 0   & 1.0 & 0   & 0 \\
0   & 0   & 0.3 & 0.7 & 0 \\
0   & 0   & 0   & 0.9 & 0.1
\end{array}\right]$$
"""

# ╔═╡ 859f3bed-8c3a-4059-b5d6-df16cb52c1e2
md"""
and the normal matrix becomes 

$$\mathbf{F}^T\,\mathbf{F} =
\left[\begin{array}{ccccc} 
0.25 & 0.25 & 0 & 0 & 0 \\
0.25 & 0.25 & 0 & 0 & 0 \\
0 & 0 & 1.09 & 0.21 & 0 \\
0 & 0 & 0.21 & 1.3 & 0.09 \\
0 & 0 & 0 & 0.09 & 0.01\end{array}\right]$$

Generally, when using an $n$-point interpolation filter for forward interpolation, the normal matrix will have $2n-1$ nonzero diagonals.
"""

# ╔═╡ 8c92517d-275a-4806-82a1-d3803372148b
md"""
In the case of B-spline interpolation with the filter

$$F(Z,\sigma) = {\frac{\sum\limits_{n=-N/2+1}^{N/2}
C_n(\sigma)\,Z^n}{B(Z)}},$$

we can express the forward interpolation in matrix form as $\mathbf{F}=\mathbf{C}\,\mathbf{B}^{-1}$, where the matrix $\mathbf{C}$ contains interpolation weights, and $\mathbf{B}^{-1}$ represents polynomial division. The matrix $\mathbf{B}$ is symmetric and positive definite. 
"""

# ╔═╡ 4d18902a-1487-40b8-b102-a912c78def53
md"""
The least-squares inverse, in this case, is

$$\begin{array}{rcl}\left(\mathbf{F}^T\,\mathbf{F}\right)^{-1}\,\mathbf{F}^T & = &
\left(\mathbf{B}^{-1}\,\mathbf{C}^T\,\mathbf{C}\,\mathbf{B}^{-1}\right)^{-1}\,\mathbf{B}^{-1}\,\mathbf{C}^T \\ & = &
\mathbf{B}\,\left(\mathbf{C}^T\,\mathbf{C}\right)^{-1}\,\mathbf{C}^T\;.\end{array}$$

In the case of $k$-th order B-spline interpolation, the normal matrix
$\mathbf{C}^T\,\mathbf{C}$ will contain $2\,k+1$ nonzero diagonals.
"""

# ╔═╡ 43d97e12-4e27-46a2-91c1-58328fbc873a
md"""
The computational cost of inverting a tridiagonal matrix using $LU$ factorization is approximately $8N$ multiplications. For a banded $N \times N$ matrix with $N_d$ nonzero diagonals, the cost increases as $O(N_d^2 N)$. As long as the number of diagonals, $N_d$, remains limited, the inversion cost stays linear in $N$ and is manageable for large-scale applications.
"""

# ╔═╡ c40a513f-c78f-4d9b-9526-08afadfcf837
md"""
### Adding model constraints

How can we incorporate model constraints to prevent the normal operator $\mathbf{F}^T\,\mathbf{F}$ from becoming singular? Let us recall the general linear model estimation recipe

$$\begin{array}{rcl}\widehat{\mathbf{m}} & = & 
\arg \min_{\mathbf{m}} \left|\mathbf{D}_n\,(\mathbf{F}\,\mathbf{m}-\mathbf{d})\right|^2+\left|\mathbf{D}_m\,\mathbf{m}\right|^2 \\ & = &
\left(\mathbf{F}^T\,\mathbf{C}_n^{-1}\,\mathbf{F} + \mathbf{C}_{m}^{-1}\right)^{-1}\,\mathbf{F}^T\,\mathbf{C}_n^{-1}\,\mathbf{d}\end{array}$$

with $\mathbf{D}_n^T\,\mathbf{D}_n=\mathbf{C}_n^{-1}$ and $\mathbf{D}_m^T\,\mathbf{D}_m=\mathbf{C}_m^{-1}$. 
"""

# ╔═╡ db8d2242-2af9-458b-a544-83b9e2a08909
md"""
In the absence of any other information about the noise covariance $\mathbf{C}_n$, it is common to select it as $\epsilon^2\,\mathbf{I}$, which represents uncorrelated normally distributed noise with variance $\epsilon^2$. Without additional information regarding the model covariance $\mathbf{C}_{m}$, it is typical to seek the simplest (smoothest) model by penalizing its derivatives. The inverse interpolation, in this case, satisfies the equation

$$\begin{array}{rcl}\widehat{\mathbf{m}} & = & \arg \min_{\mathbf{m}} \left|\mathbf{F}\,\mathbf{m}-\mathbf{d}\right|^2+\epsilon^2\,\left|\mathbf{D}_m\,\mathbf{m}\right|^2 
\\ & = & \left(\mathbf{F}^T\,\mathbf{F} + \epsilon^2\,\mathbf{D}_m^T\,\mathbf{D}_m\right)^{-1}\,\mathbf{F}^T\,\mathbf{d}\;\end{array}$$
"""

# ╔═╡ b509814d-ccaa-44a1-973c-7013c93b708a
md"""
Adding the term $\epsilon^2\,\left|\mathbf{D}_m\,\mathbf{m}\right|^2$ to the least-squares objective function is known as *Tikhonov regularization* and has various interpretations. Most importantly, it "regularizes" the problem by making the inverted matrix non-singular.
"""

# ╔═╡ 61331465-4489-48ff-9f9c-3a589acff250
md"""
For example, we can choose $\mathbf{D}_m$ as the simple difference operator, corresponding to convolution with the filter $D(Z)=1-Z$. The inverse model covariance $\mathbf{C}_m^{-1}$ in this case corresponds to convolution with the second difference $D(1/Z)\,D(Z)=-1/Z+2-Z$. 
"""

# ╔═╡ 77d92bb0-e17f-4c06-8738-f32a81018fe3
md"""
The inverted matrix transforms to

$$\begin{array}{c}\mathbf{F}^T\,\mathbf{F} +  \epsilon^2\,\mathbf{D}_m^T\,\mathbf{D}_m = \\
\left[\begin{array}{ccccc} 
0.25+2\epsilon^2 & 0.25-\epsilon^2 & 0 & 0 & 0 \\
0.25-\epsilon^2 & 0.25+2\epsilon^2 & -\epsilon^2 & 0 & 0 \\
0 & -\epsilon^2 & 1.09+2\epsilon^2 & 0.21-\epsilon^2 & 0 \\
0 & 0 & 0.21-\epsilon^2 & 1.3+2\epsilon^2 & 0.09-\epsilon^2 \\
0 & 0 & 0 & 0.09-\epsilon^2 & 0.01+2\epsilon^2
            \end{array}\right]\end{array}$$

without losing its tridiagonal structure. Different boundary conditions on the regularization filter are possible.
"""

# ╔═╡ c3c46272-8702-4eb9-b5cf-0edac80cdb6f
tmax = 80

# ╔═╡ 7dfb8f25-51a3-4a5b-a3c4-10a144e49871
function regularization1(n, ϵ)
    "derivative regularization"
    A = BandedMatrix{Float64}(undef, (n, n), (1, 1))
    A[band(0)] .= 2ϵ
    A[band(1)] .= A[band(-1)] .= -ϵ
    return A
end

# ╔═╡ 9c222fe0-fd3a-4b75-ae2c-bb8785eabb18
md"""
### Tikhonov regularization for splines

In the case of inverse B-spline interpolation, we need to apply Tikhonov regularization not to the model itself but to its spline coefficients. Let us consider a continuous function $f(x)$ and its representation as a superposition of B-spline functions

$$f(x) = \sum\limits_n c_n\,\beta(x-n)\;.$$
"""

# ╔═╡ a5c3cb97-7357-4c1c-98c6-21764a0769ff
md"""
To construct a regularization term, we can apply some differential operator $\mathbf{D}$ to this function:

$$\displaystyle \int \left|\mathbf{D} f(x)\right|^2\,dx = \int \left|\sum\limits_n c_n \mathbf{D} \beta(x-n)\right|^2\,dx = \mathbf{c}^T\,\mathbf{R}\,\mathbf{c}\;,$$

where $\mathbf{c}$ is the vector of spline coefficients. 
"""

# ╔═╡ a10ed5ba-ccee-4559-ab12-bd94443a9aff
md"""
The elements of matrix $\mathbf{R}$ are defined as

$$R_{nm} = \displaystyle \int \mathbf{D} \beta(x-n)\,\mathbf{D} \beta(x-m)\,dx\;.$$

Thanks to the stationarity of B-splines, matrix $\mathbf{R}$ is a convolution matrix: $R_{nm} = r_{n-m}$. In the Z-transform notation, multiplication by matrix $\mathbf{R}$ corresponds to multiplication by polynomial $R(Z)$.
"""

# ╔═╡ 7b4fd6f8-58d7-4540-9b56-3187bcc520cc
md"""
Recall that the case of linear interpolation corresponds to the first-order B-spline function or the triangular hat

$$\beta_1(x) = \left\{\begin{array}{rcl} 
1-|x| & \quad \mbox{for} & |x| \le 1 \\
0 & \quad \mbox{for} & |x| > 1 \end{array}\right.$$

The derivative of a hat is a step function

$$\frac{d}{dx} \beta_1(x) = \left\{\begin{array}{rcl} 
-\mbox{sign}\,{x} & \quad \mbox{for} & |x| \le 1 \\
0 & \quad \mbox{for} & |x| > 1 \end{array}\right.$$
"""

# ╔═╡ 74acec1b-f89d-4149-b5ef-cd01f1df86a1
md"""
The corresponding regularization filter

$$r_{n-m} = \displaystyle \int \frac{d \beta_1(x-n)}{dx}\,\frac{d \beta_1(x-m)}{dx}\,dx$$

leads to the convolution with the filter $R_1(Z)=-1/Z+2-Z$, which we already used in the previous section.
"""

# ╔═╡ 7377e5de-78cc-4965-80dc-d093215fc700
md"""
In the case of cubic spline interpolation, the B-spline function is
defined as

$$\beta_3(x) = \left\{\begin{array}{lcr} \displaystyle 
\left(4-6|x|^2+3 |x|^3\right)/6, & \mbox{for} & 1 > |x| \geq 0
  \\ \displaystyle (2-|x|)^3/6, & \mbox{for} & 2 > |x| \geq 1 \\ 0, &
  \mbox{elsewhere} &
\end{array}\right.$$
"""

# ╔═╡ ff377848-e9ce-4c8f-8afe-bedb1a0234d8
md"""
and has the derivative

$$\frac{d}{dx}\,\beta_3(x) = \left\{\begin{array}{lcr} \displaystyle 
(x + 2)^2/2 & \mbox{for} & -1 > x \geq -2 \\ \displaystyle 
-3/2\, x^2 - 2\,x & \mbox{for} & 0 > x \geq -1 \\ \displaystyle 
3/2\, x^2 - 2\, x & \mbox{for} & 1 > x \geq 0 \\ \displaystyle 
- (x - 2)^2/2 & \mbox{for} & 2 > x \geq 1
\\ 0, &  \mbox{elsewhere} &
\end{array}\right.$$
"""

# ╔═╡ db81acb1-6d10-4d8c-851f-5741bd0ca257
md"""

The coefficients of the corresponding filter

$$\begin{array}{rcl}R_3(Z) & = & r_3/Z^3 + r_2/Z^2
+ r_1/Z + \\ & & r_0 + r_1\,Z + r_2\,Z^2 + r_3\,Z^3\end{array}$$ 

are given by
"""

# ╔═╡ d08a3efd-f00b-456c-aa57-6696e03b12ab
md"""
$$\begin{array}{rcl}
r_0 & = & \displaystyle \int\limits_{-2}^{-1}
\left[(x + 2)^2/2\right]^2\,dx + \int\limits_{-1}^{-0}
\left[-3/2\, x^2 - 2\,x\right]^2\,dx \\ & & \displaystyle + \int\limits_{0}^{1}
\left[3/2\, x^2 - 2\, x\right]^2\,dx + \int\limits_{1}^{2}
\left[- (x - 2)^2/2\right]^2\,dx \\ = & \displaystyle \frac{2}{3}\end{array}$$ 
"""

# ╔═╡ 21d324d6-9c06-4a8b-b37f-21bc0d9acbb7
md"""
$$\begin{array}{rcl}
r_1 & = & \displaystyle \int\limits_{-2}^{-1}
\left[(x + 2)^2/2\right]\,\left[-3/2\, (x+1)^2 - 2\,(x+1)\right]\,dx
\\ \nonumber & & \displaystyle + \int\limits_{-1}^{-0}
\left[-3/2\, x^2 - 2\,x\right]\,\left[3/2\, (x+1)^2 -
  2\,(x+1)\right]\,dx 
\\ & & \displaystyle + \int\limits_{0}^{1}
\left[3/2\, x^2 - 2\, x\right]\,\left[- (x +1 - 2)^2/2\right]\,dx \\ & = &
\displaystyle -\frac{1}{8}\end{array}$$ 
"""

# ╔═╡ 283b3d96-4046-4796-9365-5005fb15cb31
md"""
$$\begin{array}{rcl}
r_2 & = & \displaystyle \int\limits_{-2}^{-1}
\left[(x + 2)^2/2\right]\,\left[3/2\, (x+2)^2 - 2\, (x+2)\right]\,dx
\\
& & \displaystyle + \int\limits_{-1}^{-0}
\left[-3/2\, x^2 - 2\,x\right]\,\left[- (x +2 - 2)^2/2\right]\,dx \\ & = & \displaystyle -
\frac{1}{5} \end{array}$$ 
"""

# ╔═╡ b4fa4552-c872-4832-aa98-42cf54c41e82
md"""
$$\begin{array}{rcl}
r_3 & = & \displaystyle \int\limits_{-2}^{-1}
\left[(x + 2)^2/2\right]\,\left[- (x +3 - 2)^2/2\right]\,dx = -
\frac{1}{120}\end{array}$$ 
"""

# ╔═╡ 53d1e336-6dcd-4d4c-9bd5-868474f3e287
md"""
The filter can be multiplied by $\epsilon^2$ and added, in matrix form, to the inverted matrix without disrupting its seven-diagonal structure.
"""

# ╔═╡ afa5b2d7-1f4a-423f-8418-863aa4e0925b
spline3(σ) = [ 
        (1 + σ*((3 - σ)*σ-3))/6,
        (4 + 3*(σ -2)*σ^2)/6,
        (1 + 3*σ*(1 + (1 - σ)*σ))/6,
        σ^3/6 ] # Cubic spline

# ╔═╡ 3e015566-088d-4cb3-a006-378933ba25d9
function regularization3(n, ϵ)
    "derivative regularization for cubic splines"
    A = BandedMatrix{Float64}(undef, (n, n), (3, 3))
    A[band(0)] .= 2ϵ/3
    A[band(1)] .= A[band(-1)] .= -ϵ/8
    A[band(2)] .= A[band(-2)] .= -ϵ/5
    A[band(3)] .= A[band(-3)] .= -ϵ/120
    return A
end

# ╔═╡ bc62169c-0965-4a1e-8673-3865fbd3a539
md"""
## Multidimensional inverse interpolation

A multidimensional interpolation problem can be converted into a series of one-dimensional problems in certain cases. For example, consider interpolating a 2-D function $f(x,y)$ sampled regularly in $x$ and $y$, and then sampling it again in new coordinates $\hat{x}$ and $\hat{y}$, which are defined by $\hat{x}=h_x(x,y)$ and $\hat{y}=h_y(x,y)$.
"""

# ╔═╡ 87396f89-5705-4ab2-9fc0-b252cf43cfa3
md"""
This problem can be solved in three steps:

1. For each $y$, interpolate from $h_y(x,y)$ to  $h_y(\hat{x},y)$ using the one-dimensional mapping provided by $h_x(x,y)$.
2. For each $y$, interpolate from $f(x,y)$ to $f(\hat{x},y)$, using the one-dimensional mapping provided by $h_x(x,y)$.
3. For each $\hat{x}$, interpolate $f(\hat{x},y)$ to $f(\hat{x},\hat{y})$ using the one-dimensional mapping provided by $h_y(\hat{x},y)$.
"""

# ╔═╡ 44421867-f751-4e49-a883-887cb9689399
md"""
The inverse interpolation can proceed in the reverse order:

1. For each $y$, interpolate $h_y(x,y)$ to $h_y(\hat{x},y)$ using the one-dimensional mapping provided by $h_x(x,y)$.
2. For each $\hat{x}$, inverse interpolate from $f(\hat{x},\hat{y})$ to $f(\hat{x},y)$ using the one-dimensional mapping provided by $h_y(\hat{x},y)$.
3. For each $y$, inverse interpolate from $f(\hat{x},y)$ to $f(x,y)$, using the one-dimensional mapping provided by $h_x(x,y)$.

A similar process can be applied in 3-D or higher dimensions.
"""

# ╔═╡ 9e9e4347-7180-4838-9dce-4b618f953073
function inverse_spline3(irreg::Array, x1::Array, x2::Array, 
                         n::Array{Int}; d=[1,1], o=[0,0]) 
    n1, n2 = n
    m1, m2 = size(x1)
    xhat = Array{eltype(x1)}(undef, m2, n1)
    for i2 in 1:m2
        xhat[i2,:] = inverse_spline3(x1[:,i2], x2[:,i2], n1, 
                                     d=d[1], o=o[1])
    end
    yhat = Array{eltype(irreg)}(undef, m2, n1)
    for i2 in 1:m2
        yhat[i2,:] = inverse_spline3(irreg[:,i2], x1[:,i2], n1, 
                                     d=d[1], o=o[1]) 
    end    
    reg = Array{eltype(irreg)}(undef, n1, n2)
    for i1 in 1:n1
        reg[i1,:] = inverse_spline3(yhat[:,i1], xhat[:,i1], n2, 
                                    d=d[2], o=o[2]) 
    end
    return reg
end

# ╔═╡ ea1adfe8-f30b-4cff-8c43-8ebf55e702b2
download("https://ahay.org/data/ctscan/circle.rsf@","circle.bin")

# ╔═╡ ca81b34e-5c82-4f14-9344-948747d86496
begin
	slice = Array{Float32}(undef, 512, 512);
	read!("circle.bin", slice)
end

# ╔═╡ 46acfc47-9fab-4d39-9169-4aebed37f3ee
download("https://ahay.org/data/ctscan/mask.rsf@","mask.bin")

# ╔═╡ 30b8a60f-075d-4f3e-a930-bc10d5396d99
begin
	mask = Array{Float32}(undef, 512, 512);
	read!("mask.bin", mask)
end

# ╔═╡ e980af8b-2746-40a6-b1a6-d770ae878b3e
plot_scan(scan, title) = heatmap(mask .* scan, title=title, c=:grays, 
                                 aspect_ratio=:equal, legend=:none, 
                                 clim=(0, 244), border=:none) 

# ╔═╡ 6a7b01aa-dab8-4e65-a144-6409a3a0d588
plot_scan(slice, "CT scan")

# ╔═╡ 63d4c0b1-e236-4dba-9a16-4b161ea8357b
function rotate(angle, n1, n2)
    # degrees to radians
    cosa, sina = cos(π * angle/180), sin(π * angle/180)
    # central point
    c1, c2 = (n1 + 1)/2, (n2 + 1)/2    
    x1 = Array{Float64}(undef, n1, n2)
    x2 = Array{Float64}(undef, n1, n2)
    for i2 in 1:n2, i1 in 1:n1
        x1[i1, i2] = c1 + (i1 - c1)*cosa + (i2 - c2)*sina
        x2[i1, i2] = c2 - (i1 - c1)*sina + (i2 - c2)*cosa
    end         
    return x1, x2
end

# ╔═╡ 3df4a421-ab88-4721-b426-876b8cd800ba
angle = 45

# ╔═╡ cb67679b-9ddf-4a25-b0cb-0cfc61237f9b
x1, x2 = rotate(angle, 512, 512);

# ╔═╡ 36655806-42b3-47ce-9f53-859eaa6ac05e
md"""
!!! assignment
    ## Task 2

    Repeat the experiment of rotating the image back and forth using `forward_linear` and `inverse_linear` applied to two-dimensional coordinate mapping.
"""

# ╔═╡ 3f5c451e-33a5-431d-9f55-d26923d0bdfb
md"""
## General inverse interpolation

In the general case of irregular space locations, the basic estimation equation remains the same. Inversion typically requires multiple iterations of the conjugate gradient method. Variations of this equation, such as the preconditioning formulation

$$\widehat{\mathbf{m}} =
\mathbf{C}_{m}\,\mathbf{F}^T\,\left(\mathbf{F}\,\mathbf{C}_{m}\,\mathbf{F}^T
    + \epsilon^2\,\mathbf{I}\right)^{-1}\,\mathbf{d}\;,$$

offer additional possibilities for constraining the estimating model and for accelerating convergence. 
"""

# ╔═╡ 5ce6e014-92d2-4bf7-89fc-b498c6639c72
md"""
### Interpolation after binning

To explore inverse interpolation in multiple dimensions, we will begin by analyzing the situation where the data are already on a regularly sampled grid, but some bins are empty.
"""

# ╔═╡ a6901eb0-0f4c-42f3-a47d-f2b781f5c794
download("https://ahay.org/data/hall/horizon.asc","horizon.asc")

# ╔═╡ 413a81b3-8fd9-47ee-9d3d-7366be0fd459
xyz = readdlm("horizon.asc") # read data from a text file

# ╔═╡ 805cfcd6-f7fb-4074-9b9b-7857e486b2d0
md"""
In this case, the forward operator $\mathbf{F}$ is a simple mask that identifies data at known locations. We can specify $\mathbf{D}_m$ as the gradient or Laplacian operator to achieve a smooth interpolation. 
"""

# ╔═╡ 5b85616d-1d0b-41d9-8faa-2e9aded6d4e8
function laplacian(m::Array, adjoint::Bool)
    n1, n2 = size(m)
    l = zeros(eltype(m), n1, n2)
    for i2 in 1:n2, i1 in 1:n1
        if i1 > 1
            if adjoint
                l[i1-1, i2] -= m[i1, i2]
                l[i1,   i2] += m[i1, i2]
            else
                l[i1, i2] += m[i1, i2] - m[i1-1, i2]
            end
        end
        if i1 < n1
            if adjoint
                l[i1+1, i2] -= m[i1, i2]
                l[i1,   i2] += m[i1, i2]
            else
                l[i1, i2] += m[i1, i2] - m[i1+1, i2]
            end
        end
        if i2 > 1
            if adjoint
                l[i1, i2-1] -= m[i1, i2]
                l[i1, i2  ] += m[i1, i2]
            else
                l[i1, i2] += m[i1, i2] - m[i1, i2-1]
            end
        end
        if i2 < n2
            if adjoint
                l[i1, i2+1] -= m[i1, i2]
                l[i1, i2  ] += m[i1, i2]
            else
                l[i1, i2] += m[i1, i2] - m[i1, i2+1]
            end
        end
    end
    return l
end

# ╔═╡ 05e7e884-00b7-4891-916d-8552ec0bc741
md"""
We will use the conjugate gradient method to minimize $\left|\mathbf{D}_m\,\mathbf{m}\right|^2$ iteratively while ensuring that $\mathbf{m}$ remains unchanged at the locations of the known data.

* **Briggs, I.C., 1974. Machine contouring using minimum curvature. Geophysics, 39(1), pp.39-48.**
"""

# ╔═╡ 565b86a0-be5f-4a9a-8fc3-c947265f50d6
function dottest(forward::Function, adjoint::Function, 
                 m::Array, d::Array)
    "Dot-product test"
    mod = similar(m); rand!(mod)
    dat = similar(d); rand!(dat)
    println(" L[m]⋅d = $(forward(mod) ⋅ dat)")
    println("L'[d]⋅m = $(adjoint(dat) ⋅ mod)")
end

# ╔═╡ 5c2936c4-80fd-402b-a348-8c19250bf8d0
function conjgrad(forward::Function, known::Array{Bool}, 
				  x0::Array, niter::Int)
    "Conjugate-gradients for minimizing |forward(x)|^2 
	 under the constraint known(x)=x0"
    x = deepcopy(x0)
    R = forward(x, false)  
    s, S = similar(x), similar(R)
    gnp = z = zero(eltype(x))
    for iter in 1:niter
        g = forward(R, true) # apply adjoint
        g[known] .= z # avoid changing x at known locations
        G = forward(g, false)
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
    end
    return x
end

# ╔═╡ e059686a-70e7-42a3-b783-4af9d59378c9
niter = 200 # number of iterations

# ╔═╡ 47bcf727-90ea-41c7-a0e9-a54dcd450a6a
md"""
!!! assignment
	## Task 3

	Modify the `conjgrad` function above to output, in addition to `x`, the residual size `(R ⋅ R)` at each iteration. To visualize convergence, plot the residual size versus the number of iterations.
"""

# ╔═╡ b2b95f29-0bb7-406a-9902-6d91bfc2addb
md"""
### Interpolation without binning

To test inverse interpolation without binning, we will return to the familiar example of rainfall interpolation in Switzerland.
"""

# ╔═╡ e7cd5663-5a66-42fc-b8c6-7ff6dca29efe
begin
	# download data files
	download("https://ahay.org/data/rain/border.rsf@","border.bin")
	download("https://ahay.org/data/rain/alldata.rsf@","alldata.bin")
	download("https://ahay.org/data/rain/obsdata.rsf@","obsdata.bin")
end

# ╔═╡ bee1b1ad-86d8-449b-9b24-e042ae31bd81
begin
	# read data
	border = Array{Float32}(undef, 2, 1289); # single-precision array
	alldata = Array{Float32}(undef, 3, 467); # single-precision array
	obsdata = Array{Float32}(undef, 3, 100); # single-precision array
	read!("border.bin", border)
	read!("alldata.bin", alldata)
	read!("obsdata.bin", obsdata)
end

# ╔═╡ b124c3b1-5993-4e69-9d2f-672b39177b26
begin
	plot(border[1,:], border[2,:], linewidth=2, label=:none)
	scatter!(alldata[1,:], alldata[2,:], ms=2, ma=0.5, label="all stations")
	scatter!(obsdata[1,:], obsdata[2,:], markershape=:utriangle, ms=4,
	    label="test stations", title="Switzerland Weather Stations")
end

# ╔═╡ 5da2d142-c401-40da-b76f-4b0714f6a80e
md"""
We will be minimizing

$\min\left( |\mathbf{F}\,\mathbf{m} - \mathbf{d}|^2 + \epsilon^2 |\mathbf{R}\,\mathbf{m}|^2\right)\;,$

where $\mathbf{d}$ is irregular data, $\mathbf{m}$ is model estimated
on a regular grid, $\mathbf{F}$ is forward interpolation from the
regular grid to irregular locations, $\epsilon$ is a scaling
parameter, and $\mathbf{R}$ is the regularization operator related to
the inverse of the assumed model covariance.
"""

# ╔═╡ 4b4cc904-5656-47fa-b0cc-c867723c93b8
function nnint(regul::Array, coord; d=[1,1], o=[0,0])
    "nearest-neighbor interpolation"
    n = size(regul)
    nd = size(coord, 2)
    irreg = Array{eltype(regul)}(undef, nd)
    for id in 1:nd
		# find nearest neighbor
        i1 = round(Int, 1 + (coord[1,id] - o[1])/d[1])
        i2 = round(Int, 1 + (coord[2,id] - o[2])/d[2])
        if 0 < i1 && i1 <= n[1] && 0 < i2 && i2 <= n[2]
            irreg[id] = regul[i1,i2]
        end
    end
    return irreg
end

# ╔═╡ ec8b0239-d4bf-4d27-bcac-4dd8aa5e355e
function nnint_adjoint(irreg::Vector{T}, coord, n::Vector{Int}; 
                       d=[1,1], o=[0,0]) where T <: Real
    "adjoint of nearest-neighbor interpolation"
    nd = size(coord, 2)
    regul = zeros(T, n[1], n[2])
    for id in 1:nd
		# find nearest neighbor
        i1 = round(Int, 1 + (coord[1,id] - o[1])/d[1])
        i2 = round(Int, 1 + (coord[2,id] - o[2])/d[2])
        if 0 < i1 && i1 <= n[1] && 0 < i2 && i2 <= n[2]
            regul[i1,i2] += irreg[id]
        end
    end
    return regul
end

# ╔═╡ 0ab64f60-5ce7-4083-852e-3ca245719b5a
rain0 = zeros(Float32, 371, 255);

# ╔═╡ edb31ad7-61e3-4cc0-9483-573e2127250f
begin
	ϵ = 0.01 # regularization parameter
	iters = 50 # number of iterations
end

# ╔═╡ dd5f97d3-cc60-4ca1-b48c-34695b87cfd2
begin
	lat = -185:185 # latitude
	lon = -127:127 # longitude
end

# ╔═╡ a639c851-3f24-4281-ab86-6cac3a9cdca0
exact = alldata[3,:];

# ╔═╡ a4957d2e-1362-48b8-9dc2-253a2af6422d
md"""
!!! assignment
	## Task 4

	Increase the number of iterations in the experiment above to achieve convergence. Then, try the following modifications to see if they improve the results:
	1. Replace nearest-neighbor interpolation with linear interpolation.
	2. Replace Laplacian minimization with gradient minimization.
"""

# ╔═╡ 62b140ea-d599-40c6-a470-f9f6cd1ce3df
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

# ╔═╡ 1fcb9e74-ab71-404e-a130-d6fc51fcdb37
function lint_adjoint(irreg::Vector{T}, coord, n::Vector{Int}; 
                      d=[1,1], o=[0,0]) where T <: Real
    "adjoint of bilinear interpolation"
    nd = size(coord, 2)
    regul = zeros(T, n[1], n[2])
    for id in 1:nd
		## !!! ADD MISSING LINES
    end
    return regul
end

# ╔═╡ 7978dbf9-7055-48c3-99d9-6844ba113401
begin
	forw2 = x -> lint(x, obsdata, o=[-185, -127])
	adjt2 = y -> lint_adjoint(y, obsdata, [371, 255], o=[-185, -127])
end

# ╔═╡ 160e741a-cfbd-4a2c-9547-af5896a89676
dottest(forw2, adjt2, rain0, obsdata[3,:])

# ╔═╡ 44410084-e668-4223-868e-5d911e855b0e
function gradient(x::Array)
	# finite-difference gradient operator
    n1, n2 = size(x)
	g = zeros(eltype(x), n1, n2, 2)
	for i1 in 1:n1-1, i2 in 1:n2-1
		g[i1, i2, 1] = x[i1+1, i2] - x[i1, i2]
		g[i1, i2, 2] = x[i1, i2+1] - x[i1, i2]
	end
	return g
end

# ╔═╡ 896f53c6-3186-4b48-abe0-57c8378dbfcd
function gradient_adjoint(g::Array)
	# finite-difference gradient operator
    n1, n2 = size(g, 1), size(g, 2)
	x = zeros(eltype(g), n1, n2)
	for i1 in 1:n1-1, i2 in 1:n2-1
		x[i1+1, i2] += g[i1, i2, 1]
		## !!! ADD MISSING LINES
	end
	return x
end

# ╔═╡ 3987bdcd-689c-4fa3-a3a8-c746eb284757
# one-line function with one-line conditional
gradient(x::Array, adj::Bool) = adj ? gradient_adjoint(x) : gradient(x)

# ╔═╡ c258f138-dcf9-4317-b881-1475c3620173
grad0 = zeros(Float32, 371, 255, 2);

# ╔═╡ acac4956-b964-42e2-82ec-504296f2dc45
dottest(gradient, gradient_adjoint, rain0, grad0)

# ╔═╡ e2c560a9-4ee6-40d8-8092-f7ce2c478b74
md"""
!!! assignment
	## Task 5

	Use your `binning` function from Task 1 to put the rainfall data into a regular grid. Then, apply the Laplacian minimization method from Task 3 to fill in the empty bins. Compare the output with previous results.
"""

# ╔═╡ 65d38c33-66b1-4d7a-8165-aac3bb1c4d44
md"""
!!! assignment
	## Bonus Task

	Let's try an alternative approach to linear estimation, where we use preconditioning operator $\mathbf{P}$ such that $\mathbf{P\,P}^T = \mathbf{C}_m$ instead of regularization operator $\mathbf{D}$ such as $\mathbf{D}^T\,\mathbf{D} = \mathbf{C}_m^{-1}$. We can use the helix transform to implement $\mathbf{P}$ as recursive filtering on a helix.
"""

# ╔═╡ c83fa056-0ae5-4663-b55e-5ae8540bbbb6
mutable struct HelixFilter
    lag::Vector{CartesianIndex}
    flt::Vector
    HelixFilter(lag,flt) = new(map(CartesianIndex,lag),flt)
end

# ╔═╡ e56a9179-11dd-4ccf-904d-d3fab072e6a5
Base.length(a::HelixFilter) = Base.length(a.lag)

# ╔═╡ 29788e97-d064-4446-b0f5-2676e028ff2e
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

# ╔═╡ e5f6fac9-e4be-47f2-a8c4-5beabfb7ba1c
plot_sketch(coord)

# ╔═╡ 773f2556-f9ec-4e40-bb86-60c2661a4895
function nnint(regul::Vector{T}, coord, d1=1, o1=0) where T <: Real
    "Nearest-neighbor interpolation"
    nm, nd = length(regul), length(coord)
    irreg = similar(coord)
    for id in 1:nd
        f = 1 + (coord[id] - o1)/d1     
        im = round(Int, f) # nearest integer      
        irreg[id] =  (0 < im && im <= nm) ? regul[im] : zero(T)
    end
    return irreg
end

# ╔═╡ f17e89e3-360c-4564-810b-9b6ae10b008c
y = nnint(x, coord)

# ╔═╡ 49e558b4-8e13-4eaa-a723-2698e7c2c808
begin
	forw = x -> nnint(x, obsdata, o=[-185, -127])
	adjt = y -> nnint_adjoint(y, obsdata, [371, 255], o=[-185, -127])
end

# ╔═╡ 322ca7e9-482c-4ba2-b5e5-6f4dc51e44a2
dottest(forw, adjt, rain0, obsdata[3,:])

# ╔═╡ 6fee81e9-ed4f-4b2a-8f8a-a366d6491105
function binning(irreg::Vector{T}, coord, n::Int, d=1, o=0) where T <: Real
    nd = length(coord)
    reg = zeros(T, n)
    count = zeros(Int, n)
    for id in 1:nd
        x = 1 + (coord[id] - o)/d
        i = round(Int, x) # nearest integer
        if i >=1 && i <= n
            reg[i] += irreg[id]
            count[i] += 1
        end
    end
    for i in 1:n
        if count[i] > 1
            reg[i] /= count[i]
        end
    end
    return reg
end

# ╔═╡ 5dff593e-b719-405c-9c7a-faa71c6d5888
binning(y, coord, 5)

# ╔═╡ 06339a14-2b07-42e7-8826-81725beea046
function forward_linear(reg::Vector{T}, coord; 
                        d=1, o=0) where T <: Real
    "forward linear interpolation"
    n = length(reg)
    nd = length(coord)
    irreg = zeros(T, nd)
    for id in 1:nd
        x = (coord[id] - o)/d
        i = floor(Int, x) # nearest integer less or equal to x
        x -= i
        w = [1-x, x]
        j1, j2 = max(1,1-i), min(2,n-i)
        for j in j1:j2
             irreg[id] += w[j] * reg[i+j]
        end
    end
    return irreg
end 

# ╔═╡ b273bb4b-1151-4a16-a052-4ed9b9e81dfe
y2 = forward_linear(float.(x), coord)

# ╔═╡ 5de9be08-a300-4252-8972-17e9c8cb0428
function chirp(x)
    n = length(x)
    image = Array{Float32}(undef,n,n)
    for i in 1:n, j in 1:n
        x1, x2 = x[i], x[j]
        f = (0.25*x1^2 + x2^2)/(0.25*tmax)^2
        image[j,i] = 0.5*cos(8*f)*exp(-f)
    end
    return image
end

# ╔═╡ c7bbebcf-a0cd-40b3-88e4-b6615753a61c
begin
	xc(n) = [((i-1)/(n-1) - 1/2)*tmax for i in 1:n]
	xdense = xc(500)
	xspars = xc(50)
	dense = chirp(xdense)
	spars = chirp(xspars)
	dx=xspars[2]-xspars[1];
	x0=xspars[1];
end

# ╔═╡ 3b65f7aa-df9b-4221-aee1-b52f335ebd12
begin
	p1 = heatmap(xdense, xdense, dense, color=:grays, title="Ideal")
	p2 = heatmap(xspars, xspars, spars, color=:grays, title="Decimated")
	plot(p1, p2, layout=(1, 2))
end

# ╔═╡ 0b2b771d-91d6-415e-90d1-9a56b075f60a
begin
	in1 = mapslices(x -> forward_linear(x, xdense, d=dx, o=x0), 
	                spars; dims=1);
	in2 = mapslices(x -> forward_linear(x, xdense, d=dx, o=x0), 
	                in1; dims=2);
	error = in2 - dense;
end

# ╔═╡ 762eaa20-9ce9-4326-a73c-0dcdf7f6f29b
plot(xdense, error[:,250], ylim=[-0.15,0.15], linewidth=2,
     title="Linear Interpolation Error", label=:none)

# ╔═╡ 30b55dd7-1f94-4fa7-a044-de2d37df9fc4
begin
	pl1 = heatmap(xdense, xdense, in2, color=:grays, 
	             clim=(-0.5, 0.5), title="Linear Interpolation")
	pl2 = heatmap(xdense, xdense, error, color=:grays, 
	             clim=(-0.5, 0.5), title="Interpolation Error")
	plot(pl1, pl2, layout=(1, 2))
end

# ╔═╡ cd5264be-92df-4c2a-8dc6-10f7035b3976
function inverse_linear(irreg::Vector{T}, coord, n::Int; 
                        d=1, o=0, ϵ=0.01) where T <: Real
    nd = length(coord)
    reg = zeros(T, n)
    A = regularization1(n, ϵ)
    for id in 1:nd
        x = (coord[id] - o)/d
        i = floor(Int, x) # nearest integer less or equal to x
        x -= i
        w = [1-x, x]
        j1, j2 = max(1,1-i), min(2,n-i)
        for j in j1:j2
            reg[i+j] += w[j] * irreg[id]
            A[i+j, i+j] += w[j] * w[j]
            for k in 1:j2-j
                A[i+j, i+j+k] += w[j] * w[j+k]
                A[i+j+k, i+j] += w[j] * w[j+k]
            end
        end
    end
    reg = A \ reg
    return reg
end 

# ╔═╡ 39c01aa5-eeb1-47c7-84ad-f8a18ec0c384
inverse_linear(y, coord, 5)

# ╔═╡ 98c83171-01ac-4d86-bbdf-09fa11ae1487
inverse_linear(y, coord, 5, ϵ=1e-6)

# ╔═╡ 01be592f-c1d7-4ef4-97ec-b0db853252f8
function forward_spline3(reg::Vector{T}, coord; d=1, o=0) where T <: Real
    "forward interpolation"
    n = length(reg)
    B = BandedMatrix{Float64}(undef, (n, n), (1, 1))
    B[band(0)] .= 2/3
    B[band(1)] .= B[band(-1)] .= 1/6
    c = B \ reg # prefiltering to get spline coefficients
    nd = length(coord)
    irreg = zeros(T, nd)
    for id in 1:nd
        x = (coord[id] - o)/d - 1
        i = floor(Int, x) # nearest integer less or equal to x
        w = spline3(x-i)
        j1, j2 = max(1,1-i), min(4,n-i)
        for j in j1:j2
             irreg[id] += w[j] * c[i+j]
        end
    end
    return irreg
end 

# ╔═╡ 22b5e4cf-fd8e-4ecf-975a-138f767cf7fb
function trim!(x::Vector{T}, diag::Vector, d) where T <: Real
    "trim ends"
    n = length(x)
    i = 1
    while i < n && diag[i] == d
        i += 1
    end
    x[1:i-1] .= x[i]
    i = n
    while i > 1 && diag[i] == d
        i -= 1
    end
    x[i+1:n] .= x[i]
end

# ╔═╡ 7afb4942-c3a7-4665-a854-52a2d4c85d03
function postfilter3(c::Vector{T}) where T <: Real
    "convert cubic spline coefficients to function samples"
    n = length(c)
    x = similar(c)
    b0, b1 = 2/3, 1/6
    x[1] = b0*c[1] + b1*c[2] 
    for i in 2:n-1
        x[i] = b0*c[i] + b1*(c[i - 1] +  c[i + 1])
    end
    x[n] = b0*c[n] + b1*c[n-1] 
    return x
end

# ╔═╡ ee6b157b-7c77-4f41-ad53-f9249a3ac1bd
function inverse_spline3(irreg::Vector{T}, coord, n::Int; 
                         d=1, o=0, ϵ=0.01) where T <: Real
    nd = length(coord)
    reg = zeros(T, n)
    A = regularization3(n, ϵ)
    for id in 1:nd
        x = (coord[id] - o)/d - 1
        i = floor(Int, x) # nearest integer less or equal to x
        w = spline3(x-i)
        j1, j2 = max(1,1-i), min(4,n-i)
        for j in j1:j2
             reg[i+j] += w[j] * irreg[id]
             A[i+j, i+j] += w[j] * w[j]
             for k in 1:j2-j
                A[i+j, i+j+k] += w[j] * w[j+k]
                A[i+j+k, i+j] += w[j] * w[j+k]
            end
        end
    end
    c = A \ reg
    trim!(c, A[band(0)], 2ϵ/3)
    reg = postfilter3(c) # postfiltering from spline coefficients
    trim!(reg, A[band(0)], 2ϵ/3)
    return reg
end

# ╔═╡ 5db4181f-cec2-4f2a-9d3a-bfffafc56929
function forward_spline3(reg::Array, x1::Array, x2::Array; 
                         d=[1,1], o=[0,0]) 
    n1, n2 = size(reg)
    m1, m2 = size(x1)
    xhat = Array{eltype(x1)}(undef, m2, n1)
    for i2 in 1:m2
        xhat[i2,:] = inverse_spline3(x1[:,i2], x2[:,i2], n1, 
                                     d=d[1], o=o[1])
    end
    yhat = Array{eltype(reg)}(undef, m2, n1)
    for i1 in 1:n1
        yhat[:,i1] = forward_spline3(reg[i1,:], xhat[:,i1], 
                                     d=d[2], o=o[2]) 
    end
    irreg = similar(x1)
    for i2 in 1:m2
        irreg[:,i2] = forward_spline3(yhat[i2,:], x1[:,i2], 
                                      d=d[1], o=o[1]) 
    end
    return irreg
end

# ╔═╡ 10b761bb-8872-4650-b81a-87109d98bc35
begin
	ins1 = mapslices(x -> forward_spline3(x, xdense, d=dx, o=x0), 
	                spars; dims=1);
	ins2 = mapslices(x -> forward_spline3(x, xdense, d=dx, o=x0), 
	                in1; dims=2);
	serror = ins2 - dense;
end

# ╔═╡ 541f3522-8534-4a4e-8a85-385f3d051c28
plot(xdense, serror[:,250], ylim=[-0.15,0.15], linewidth=2,
     title="Cubic Spline Interpolation Error", label=:none)

# ╔═╡ 8e34d5ba-c8fe-4e69-a941-2a3ae76629a3
begin
	ps1 = heatmap(xdense, xdense, ins2, color=:grays, 
	             clim=(-0.5, 0.5), title="Spline Interpolation")
	ps2 = heatmap(xdense, xdense, serror, color=:grays, 
	             clim=(-0.5, 0.5), title="Interpolation Error")
	plot(ps1, ps2, layout=(1, 2))
end

# ╔═╡ 9d7b343c-bd3d-4c59-b221-3fd914c70d04
rotated = forward_spline3(slice, x1, x2);

# ╔═╡ 214afff1-73b5-47bd-b834-fb8e9c65e0c0
plot_scan(rotated, "Rotated by $(angle)°")

# ╔═╡ 5692ccb3-1f2f-4522-9fc9-a0c11d9a4524
back = inverse_spline3(rotated, x1, x2, [512, 512]);

# ╔═╡ 05f3aed0-20c6-4a0f-a1c4-dc9986b84e7b
plot_scan(back, "Rotated Back")

# ╔═╡ c06d5349-590f-45dd-a11f-cb81a5395f29
plot_scan(back-slice, "Error")

# ╔═╡ 52ca71b2-cc2b-473f-82d5-bd2994574a2f
begin
	n1, n2 = 196, 291
	iline = xyz[1:n1:end, 1]
	xline = xyz[1:n1, 2]
	horizon=reshape(xyz[:, 3], (n1, n2))
	# subtract mean
	mean = sum(horizon)/length(horizon)
    horizon .-= mean
end

# ╔═╡ 3983792b-3f98-4e92-a841-b960fa854647
plot_horizon(map, title) = heatmap(iline, xline, map, title=title, 
    xlabel=L"$x$ (m)", ylabel=L"$y$ (m)", clim=(-14,14))

# ╔═╡ ef0f6740-da15-4e67-a00a-fa12ed302047
plot_horizon(horizon, "Seismic Horizon")

# ╔═╡ 974f49db-5357-45e4-bc20-e2a9640f9983
begin
	# cut holes
	horizon2 = deepcopy(horizon)
	horizon2[126:147, 151:172] .= 0
	horizon2[151:172, 51:72] .= 0
	horizon2[76:97, 126:147] .= 0
end

# ╔═╡ b2f80156-395f-4e52-9810-52c3e02918e9
plot_horizon(horizon2, "Input")

# ╔═╡ 9c8ef0e7-6d51-4780-94c9-7be5fb6768e7
begin
	xmask = ones(Bool, size(horizon))
	xmask[126:147, 151:172] .= false
	xmask[151:172, 51:72] .= false
	xmask[76:97, 126:147] .= false
end

# ╔═╡ 463332a5-d0cb-48df-b81c-a4446e6e46e8
plot_horizon(xmask*14, "Space Mask")

# ╔═╡ d84de5fe-b69f-41a7-85c1-1ce3fa014142
dottest(x -> laplacian(x, false), x -> laplacian(x, true), horizon, horizon)

# ╔═╡ 3612e680-5551-426e-b81e-efe2548e89ca
function conjgrad(forward::Function, adjoint::Function, regularization::Function, 
                  d::Array, x0::Array, ϵ::Real, niter::Int)
    "Conjugate-gradients for minimizing |forward(x)-d|^2 + ϵ^2*|regul(x)|^2"
    x = deepcopy(x0)
    R1 = forward(x) - d
    R2 = ϵ*regularization(x, false)
    nd = length(d)
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

# ╔═╡ c6054383-4c64-4314-9626-62fa9805b7c9
filled = conjgrad(laplacian, xmask, horizon2, niter);

# ╔═╡ d2883fcc-e2d5-4cf1-9248-ebcfe95241e7
plot_horizon(filled, "Filled")

# ╔═╡ 54545940-fdeb-4fe5-9dd2-5c6c3a459ccc
map1 = conjgrad(forw, adjt, laplacian, obsdata[3,:], rain0, ϵ, iters);

# ╔═╡ 5635fc15-f89e-410a-b7db-86e45cd6ec20
heatmap(lat, lon, map1', title="Laplacian Regularization ($iters iterations)", cmap=:viridis)

# ╔═╡ 65a61bab-d011-429a-ad6a-8531d027200a
predict = nnint(map1, alldata, o=[-185, -127]);

# ╔═╡ 94ffb10c-9fe6-4826-a956-bbc6b6b4ee59
begin
	lim = [-10,600]
	# correlation coefficient
	cc = (exact ⋅ predict)/(sqrt(exact ⋅ exact) * sqrt(predict ⋅ predict))
	scatter(exact, predict, xlabel="True", ylabel="Predicted", 
		title="Gradient Regularization, cc=$(Float16(cc))",
	    aspect_ratio=:equal, xlim=lim, ylim=lim, label=:none)
	plot!(lim, lim, label=:none)
end

# ╔═╡ 9c4c51ca-978d-4f45-a710-5c80b5841cc7
# conjugate-gradients with preconditioning
function conjgrad_precon(forward::Function, adjoint::Function,
						 precon::Function, 
						 d::Array, x0::Array, ϵ::Real, niter::Int)
    "Conjugate-gradients for minimizing |forward(precon(x))-d|^2 + ϵ^2*|x|^2"
    x = deepcopy(x0)
    R1 = forward(precon(x, false)) - d
    R2 = ϵ*x
    nd = length(d)
    s, S1, S2 = similar(x), similar(R1), similar(R2)
    gnp = zero(eltype(x))
    for iter in 1:niter
        g = precon(adjoint(R1), true) + ϵ*R2 # block adjoint
        G1 = forward(precon(g, false))
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
    return precon(x, false)
end

# ╔═╡ d108b544-aae3-48e2-ba01-b500ab608f5e
function helix(a::HelixFilter, ci::CartesianIndices)
    "convert helix lags to 1-D for a given grid"
    # middle of the grid
    mid = CartesianIndex(Tuple(last(ci)) .÷ 2)
    # helix index of middle
    hmid = LinearIndices(ci)[mid]
    # from Cartesian shift to helix shift
    return LinearIndices(ci)[map(x -> x + mid, a.lag)] .- hmid
end

# ╔═╡ 030cc483-dcf9-4c14-a3f0-269aedd92fbb
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

# ╔═╡ 7dba3bab-b250-462b-ba8a-65d4a8e55df1
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

# ╔═╡ 528aa1ac-a026-4ef6-a455-14de2c61b1c1
function hwilson(au0::Real, auto::HelixFilter, lag, niter=10, pad=5)
    "Wilson-Burg spectral factorization on a helix"
    # initialize filter
    na = length(lag)
    T = eltype(auto.flt)
    a = HelixFilter(lag,zeros(T,na))
    grid = Tuple(pad*(maximum(a.lag)-minimum(a.lag)))
    au = zeros(T,grid)
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
        ϵ = 0
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

# ╔═╡ 5d945c97-2f63-456d-a122-fa5a82b639c3
begin
	# define helical derivative
	hlaplacian = HelixFilter([(1,0),(0,1)],[-1.0,-1.0])
	hlag = vcat([(x,0) for x in 1:10],[(x,1) for x in -10:0])
	helder = hwilson(4.0, hlaplacian, hlag)
	# make sure the filter removes DC
    helder.flt /= sum(-helder.flt)
end

# ╔═╡ 0428aa0b-6dfc-426a-8134-dc28724d7392
precon(x, adj) = hrecursive(x, helder, adj)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUIExtra = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
BandedMatrices = "~1.7.5"
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
project_hash = "1408b022f7c6ef9ee93e966fac5bcb3f9db9f7f9"

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
# ╟─95ac1be7-0da0-40f9-a2dd-dd01dd3f4867
# ╟─6f8f1767-8aea-4719-8428-61a973e2417e
# ╟─753d25ab-f93a-4f20-9a5d-ec83c798f550
# ╟─ed4b2c7b-e54b-4dfb-8c69-18e7a2f94311
# ╟─9142114d-aab9-4d2c-aebf-c78d74ad60c6
# ╠═7e07cef4-7234-4532-bbc7-f7c36438fd52
# ╟─29788e97-d064-4446-b0f5-2676e028ff2e
# ╠═ead28bad-dfb6-4b43-b144-fe3696133b16
# ╠═e5f6fac9-e4be-47f2-a8c4-5beabfb7ba1c
# ╟─510b8776-83bd-4258-aa4a-02f7d0f2e74c
# ╟─637b2fe3-bc7b-4a39-a122-60153938b6ed
# ╟─f507331e-6bf6-4696-a52b-fc53f0390e05
# ╟─19005a1a-0aa3-43b2-b48f-4e5d3b8cd48a
# ╠═773f2556-f9ec-4e40-bb86-60c2661a4895
# ╠═6fee81e9-ed4f-4b2a-8f8a-a366d6491105
# ╠═ee067fde-a109-4080-92bc-9f1bf849af02
# ╠═f17e89e3-360c-4564-810b-9b6ae10b008c
# ╠═5dff593e-b719-405c-9c7a-faa71c6d5888
# ╟─ba61e926-b497-4c4f-a533-09d1552041ae
# ╠═cfea9f3e-097a-4778-8b83-242b817cb2b5
# ╟─429714dc-576b-47a2-87f7-f5f99caa7379
# ╟─859f3bed-8c3a-4059-b5d6-df16cb52c1e2
# ╟─8c92517d-275a-4806-82a1-d3803372148b
# ╟─4d18902a-1487-40b8-b102-a912c78def53
# ╟─43d97e12-4e27-46a2-91c1-58328fbc873a
# ╟─c40a513f-c78f-4d9b-9526-08afadfcf837
# ╟─db8d2242-2af9-458b-a544-83b9e2a08909
# ╟─b509814d-ccaa-44a1-973c-7013c93b708a
# ╟─61331465-4489-48ff-9f9c-3a589acff250
# ╟─77d92bb0-e17f-4c06-8738-f32a81018fe3
# ╠═06339a14-2b07-42e7-8826-81725beea046
# ╠═b273bb4b-1151-4a16-a052-4ed9b9e81dfe
# ╠═c3c46272-8702-4eb9-b5cf-0edac80cdb6f
# ╠═5de9be08-a300-4252-8972-17e9c8cb0428
# ╠═c7bbebcf-a0cd-40b3-88e4-b6615753a61c
# ╠═3b65f7aa-df9b-4221-aee1-b52f335ebd12
# ╠═0b2b771d-91d6-415e-90d1-9a56b075f60a
# ╠═762eaa20-9ce9-4326-a73c-0dcdf7f6f29b
# ╠═30b55dd7-1f94-4fa7-a044-de2d37df9fc4
# ╠═f2937b56-1b31-4710-9433-fc67a4326f93
# ╠═7dfb8f25-51a3-4a5b-a3c4-10a144e49871
# ╠═cd5264be-92df-4c2a-8dc6-10f7035b3976
# ╠═39c01aa5-eeb1-47c7-84ad-f8a18ec0c384
# ╠═98c83171-01ac-4d86-bbdf-09fa11ae1487
# ╟─9c222fe0-fd3a-4b75-ae2c-bb8785eabb18
# ╟─a5c3cb97-7357-4c1c-98c6-21764a0769ff
# ╟─a10ed5ba-ccee-4559-ab12-bd94443a9aff
# ╟─7b4fd6f8-58d7-4540-9b56-3187bcc520cc
# ╟─74acec1b-f89d-4149-b5ef-cd01f1df86a1
# ╟─7377e5de-78cc-4965-80dc-d093215fc700
# ╟─ff377848-e9ce-4c8f-8afe-bedb1a0234d8
# ╟─db81acb1-6d10-4d8c-851f-5741bd0ca257
# ╟─d08a3efd-f00b-456c-aa57-6696e03b12ab
# ╟─21d324d6-9c06-4a8b-b37f-21bc0d9acbb7
# ╟─283b3d96-4046-4796-9365-5005fb15cb31
# ╟─b4fa4552-c872-4832-aa98-42cf54c41e82
# ╟─53d1e336-6dcd-4d4c-9bd5-868474f3e287
# ╠═afa5b2d7-1f4a-423f-8418-863aa4e0925b
# ╠═01be592f-c1d7-4ef4-97ec-b0db853252f8
# ╠═10b761bb-8872-4650-b81a-87109d98bc35
# ╠═541f3522-8534-4a4e-8a85-385f3d051c28
# ╠═8e34d5ba-c8fe-4e69-a941-2a3ae76629a3
# ╠═3e015566-088d-4cb3-a006-378933ba25d9
# ╠═22b5e4cf-fd8e-4ecf-975a-138f767cf7fb
# ╠═7afb4942-c3a7-4665-a854-52a2d4c85d03
# ╠═ee6b157b-7c77-4f41-ad53-f9249a3ac1bd
# ╟─bc62169c-0965-4a1e-8673-3865fbd3a539
# ╟─87396f89-5705-4ab2-9fc0-b252cf43cfa3
# ╟─44421867-f751-4e49-a883-887cb9689399
# ╠═5db4181f-cec2-4f2a-9d3a-bfffafc56929
# ╠═9e9e4347-7180-4838-9dce-4b618f953073
# ╠═ea1adfe8-f30b-4cff-8c43-8ebf55e702b2
# ╠═ca81b34e-5c82-4f14-9344-948747d86496
# ╠═46acfc47-9fab-4d39-9169-4aebed37f3ee
# ╠═30b8a60f-075d-4f3e-a930-bc10d5396d99
# ╠═e980af8b-2746-40a6-b1a6-d770ae878b3e
# ╠═6a7b01aa-dab8-4e65-a144-6409a3a0d588
# ╠═63d4c0b1-e236-4dba-9a16-4b161ea8357b
# ╠═3df4a421-ab88-4721-b426-876b8cd800ba
# ╠═cb67679b-9ddf-4a25-b0cb-0cfc61237f9b
# ╠═9d7b343c-bd3d-4c59-b221-3fd914c70d04
# ╠═214afff1-73b5-47bd-b834-fb8e9c65e0c0
# ╠═5692ccb3-1f2f-4522-9fc9-a0c11d9a4524
# ╠═05f3aed0-20c6-4a0f-a1c4-dc9986b84e7b
# ╠═c06d5349-590f-45dd-a11f-cb81a5395f29
# ╟─36655806-42b3-47ce-9f53-859eaa6ac05e
# ╟─3f5c451e-33a5-431d-9f55-d26923d0bdfb
# ╟─5ce6e014-92d2-4bf7-89fc-b498c6639c72
# ╠═4720a77f-b7c1-4e6e-ab5f-fbc133b9a4e2
# ╠═a6901eb0-0f4c-42f3-a47d-f2b781f5c794
# ╠═413a81b3-8fd9-47ee-9d3d-7366be0fd459
# ╠═52ca71b2-cc2b-473f-82d5-bd2994574a2f
# ╠═3983792b-3f98-4e92-a841-b960fa854647
# ╠═ef0f6740-da15-4e67-a00a-fa12ed302047
# ╠═974f49db-5357-45e4-bc20-e2a9640f9983
# ╠═b2f80156-395f-4e52-9810-52c3e02918e9
# ╠═9c8ef0e7-6d51-4780-94c9-7be5fb6768e7
# ╠═463332a5-d0cb-48df-b81c-a4446e6e46e8
# ╟─805cfcd6-f7fb-4074-9b9b-7857e486b2d0
# ╠═5b85616d-1d0b-41d9-8faa-2e9aded6d4e8
# ╟─05e7e884-00b7-4891-916d-8552ec0bc741
# ╠═31279552-8984-4017-affa-404f26064e32
# ╠═565b86a0-be5f-4a9a-8fc3-c947265f50d6
# ╠═d84de5fe-b69f-41a7-85c1-1ce3fa014142
# ╠═5c2936c4-80fd-402b-a348-8c19250bf8d0
# ╠═e059686a-70e7-42a3-b783-4af9d59378c9
# ╠═c6054383-4c64-4314-9626-62fa9805b7c9
# ╠═d2883fcc-e2d5-4cf1-9248-ebcfe95241e7
# ╟─47bcf727-90ea-41c7-a0e9-a54dcd450a6a
# ╟─b2b95f29-0bb7-406a-9902-6d91bfc2addb
# ╠═e7cd5663-5a66-42fc-b8c6-7ff6dca29efe
# ╠═bee1b1ad-86d8-449b-9b24-e042ae31bd81
# ╠═b124c3b1-5993-4e69-9d2f-672b39177b26
# ╟─5da2d142-c401-40da-b76f-4b0714f6a80e
# ╠═3612e680-5551-426e-b81e-efe2548e89ca
# ╠═4b4cc904-5656-47fa-b0cc-c867723c93b8
# ╠═ec8b0239-d4bf-4d27-bcac-4dd8aa5e355e
# ╠═49e558b4-8e13-4eaa-a723-2698e7c2c808
# ╠═0ab64f60-5ce7-4083-852e-3ca245719b5a
# ╠═322ca7e9-482c-4ba2-b5e5-6f4dc51e44a2
# ╠═edb31ad7-61e3-4cc0-9483-573e2127250f
# ╠═54545940-fdeb-4fe5-9dd2-5c6c3a459ccc
# ╠═dd5f97d3-cc60-4ca1-b48c-34695b87cfd2
# ╠═5635fc15-f89e-410a-b7db-86e45cd6ec20
# ╠═a639c851-3f24-4281-ab86-6cac3a9cdca0
# ╠═65a61bab-d011-429a-ad6a-8531d027200a
# ╠═94ffb10c-9fe6-4826-a956-bbc6b6b4ee59
# ╟─a4957d2e-1362-48b8-9dc2-253a2af6422d
# ╠═62b140ea-d599-40c6-a470-f9f6cd1ce3df
# ╠═1fcb9e74-ab71-404e-a130-d6fc51fcdb37
# ╠═7978dbf9-7055-48c3-99d9-6844ba113401
# ╠═160e741a-cfbd-4a2c-9547-af5896a89676
# ╠═44410084-e668-4223-868e-5d911e855b0e
# ╠═896f53c6-3186-4b48-abe0-57c8378dbfcd
# ╠═3987bdcd-689c-4fa3-a3a8-c746eb284757
# ╠═c258f138-dcf9-4317-b881-1475c3620173
# ╠═acac4956-b964-42e2-82ec-504296f2dc45
# ╟─e2c560a9-4ee6-40d8-8092-f7ce2c478b74
# ╟─65d38c33-66b1-4d7a-8165-aac3bb1c4d44
# ╠═9c4c51ca-978d-4f45-a710-5c80b5841cc7
# ╠═c83fa056-0ae5-4663-b55e-5ae8540bbbb6
# ╠═e56a9179-11dd-4ccf-904d-d3fab072e6a5
# ╠═d108b544-aae3-48e2-ba01-b500ab608f5e
# ╠═030cc483-dcf9-4c14-a3f0-269aedd92fbb
# ╠═7dba3bab-b250-462b-ba8a-65d4a8e55df1
# ╠═528aa1ac-a026-4ef6-a455-14de2c61b1c1
# ╠═5d945c97-2f63-456d-a122-fa5a82b639c3
# ╠═0428aa0b-6dfc-426a-8134-dc28724d7392
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
