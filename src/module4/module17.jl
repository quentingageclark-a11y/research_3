### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ f4712ba3-c259-4782-8656-ec8ee53d701c
begin
	using PlutoUI
	TableOfContents()
end

# ╔═╡ 01ba7a19-4f88-472a-9679-39af1a109b0d
using Plots

# ╔═╡ 280f7209-6291-43aa-926e-918497cf9d24
using LaTeXStrings

# ╔═╡ f201996f-84a7-4f7d-ab32-6647515025f7
using FastMarching

# ╔═╡ b99b18c8-bfb5-46f9-b733-1ff817d1f1ae
using DataFrames, CSV

# ╔═╡ 8bd6f886-ffca-417e-99cd-ac98f5a0460b
import HTTP

# ╔═╡ c6235c5d-ecca-421f-bbcf-65af557b9758
let
    notebook_dir = @__DIR__
    toc_path = joinpath(notebook_dir, "TOC_Notebook.jl")
    
    Markdown.parse("[⬅️ Back to Table of Contents](./open?path=$(HTTP.escapeuri(toc_path)))")
end

# ╔═╡ 02882467-033a-42a7-97e6-02d8b6e309b7
md"""
# Distance Computations


In various spatial data analysis applications, it is helpful to compute distance functions from a set of data points. Computing distances on a grid involves solving the eikonal equation numerically, typically using finite-difference methods.
"""

# ╔═╡ de2cae4e-d7bf-4484-96d0-4e3ffa5afd0b
md"""
## Eikonal equation

The distance function $D(\mathbf{x})$ satisfies the eikonal equation, a first-order nonlinear partial differential equation given by

$$\nabla D \cdot \nabla D = 1$$

with the boundary conditions $D(\mathbf{x}_k)=0$ at the source points
$\mathbf{x}_k$, $k=1,2,\cdots$ 

This is a special case of the more general eikonal equation

$$\nabla T \cdot \nabla T = S^2(\mathbf{x})$$

for the isotropic traveltime function $T(\mathbf{x})$ corresponding to
the slowness $S(\mathbf{x})$. The eikonal equation captures the geometry of the propagating wavefront. Wavefronts correspond to level sets: surfaces where $T(\mathbf{x})=t$.

One way to solve the eikonal equation is to use *ray tracing*, which tracks the trajectories of individual points on the propagating wavefront. To do so, it is convenient to rewrite the eikonal equation in the form

$$H(\mathbf{x},\mathbf{p}) = \mathbf{p} \cdot \mathbf{p} - 
  S^2(\mathbf{x}) = 0\;,$$

where the vector $\mathbf{p}$ denotes $\nabla T$. According to the eikonal equation, the function $H(\mathbf{x},\mathbf{p})$ is zero everywhere and remains constant along rays (the trajectories of individual wavefront points). Let $\sigma$ denote a variable moving along a ray. Then the differential $d\,H/d\,\sigma$ should be zero. Assuming that $S(\mathbf{x})$ is differentiable and differentiating with respect to $\sigma$, we obtain

$${\frac{d\,H}{d\,\sigma}} = 
  2\,\mathbf{p} \cdot {\frac{d\,\mathbf{p}}{d\,\sigma}} -
  2\,S(\mathbf{x})\,\nabla S \cdot {\frac{d\,\mathbf{x}}{d\,\sigma}} = 0\;.$$
"""

# ╔═╡ 0f1cb87f-60f4-464b-9ead-71940e81132b
md"""
Note that we have not specified the precise meaning of $\sigma$. For now, it is just a parameter that varies monotonically along each ray. The choice of $\sigma$ becomes constrained when we split the equation into a system of ordinary differential equations. This procedure is known in mathematics as the Hamilton-Jacobi theory, after the work of Hamilton and Jacobi in the 19th century. For example, we can split it as follows:

$$\begin{array}{rcl}\displaystyle \frac{d\,\mathbf{x}}{d\,\sigma} & = & \mathbf{p}\;, \\
  \displaystyle \frac{d\,\mathbf{p}}{d\,\sigma} & = & S(\mathbf{x})\,\nabla S\;.\end{array}$$

Note that the two equations can be combined into a single second-order equation for the ray trajectory $\mathbf{x}(\sigma)$:

$${\frac{d^2\,\mathbf{x}}{d\,\sigma^2}} =  S(\mathbf{x})\,\nabla S$$

The $\sigma$ variable now has a defined meaning and physical dimensions: distance squared per unit time. Starting from an initial point $\mathbf{x}(0)=\mathbf{x}_0$ and an initial direction vector $\mathbf{p}(0)=\mathbf{p}_0$, we can trace the ray trajectory $\mathbf{x}(\sigma)$. Repeating this process for every point on the wavefront traces the wavefront's motion.

If the slowness $S(\mathbf{x})$ is constant, $\mathbf{p}$ does not depend on $\sigma$, and all rays are straight lines described by equation

$$\mathbf{x}(\sigma) = \mathbf{x}_0 + \mathbf{p}_0\,\sigma\;.$$
"""

# ╔═╡ 0a821bd1-80e0-4c16-8e71-f8eb46cd9563
md"""
How does traveltime vary along a ray? According to the chain rule, the derivative of any scalar function $f(\mathbf{x})$ along the $\sigma$ direction is given by

$$\frac{d\,f}{d\,\sigma} = \nabla f \cdot
\frac{d\,\mathbf{x}}{d\,\sigma}\;.$$

Similarly, for the traveltime function $T(\mathbf{x})$,

$${\frac{d\,T}{d\,\sigma}} = \nabla T \cdot
  {\frac{d\,\mathbf{x}}{d\,\sigma}}$$

or, using the notation $\mathbf{p} = \nabla T$ and the ray tracing equations,

$${\frac{d\,T}{d\,\sigma}}
  = \mathbf{p} \cdot \mathbf{p} =
  S^2(\mathbf{x})\;.$$

We can track wavefront changes along individual rays by solving the last equation together with the ray-tracing equations.
"""

# ╔═╡ 95f77583-64c6-4864-a1db-da6cd99c13ee
md"""
In the constant-velocity case $S(\mathbf{x_0})=S_0$, the traveltime changes linearly as

$$T(\sigma) = S_0^2\,\sigma\;.$$

Correspondingly, the traveltime between two end points $\mathbf{x}_0$ and $\mathbf{x}_1$ is proportional to the distance

$$\widehat{T}(\mathbf{x}_0,\mathbf{x}_1) = \left|\mathbf{x}_1 - \mathbf{x}_0\right|\,S_0\;.$$
"""

# ╔═╡ 16b57010-35b7-4353-90e4-2d523c794e60
md"""
## Connection with Fermat's principle

Let us consider a ray connecting points $\mathbf{x}_0$ and $\mathbf{x}_1$, along with other arbitrary trajectories $\mathbf{x}(\sigma)$ such that $\mathbf{x}(0)=\mathbf{x}_0$ and $\mathbf{x}(\hat{\sigma})=\mathbf{x}_1$. Further, we assume that there is a functional of the trajectory

$$F[\mathbf{x}(\sigma)] = \int\limits_{0}^{\hat{\sigma}} L\left[\mathbf{x}(\sigma),\mathbf{p}(\sigma),\sigma\right]\,d \sigma$$

which becomes stationary when the trajectory is a ray. Here $L$ is an unspecified function that may depend not only on the shape of the trajectory $\mathbf{x}(\sigma)$ but also on its local slope $\mathbf{x}'(\sigma)$, denoted by $\mathbf{p}(\sigma)$. Being stationary means that we can add a perturbation to the trajectory $\mathbf{x}_\epsilon(\sigma) = \mathbf{x}(\sigma) + \epsilon\,\mathbf{h}(\sigma)$, and provided that $\mathbf{h}(0)=0$ and $\mathbf{h}(\hat{\sigma})=0$, the corresponding perturbation of $F$ will be zero at $\epsilon=0$ if $\mathbf{x}(\sigma)$ is a ray. Mathematically,

$$0 = \left.\frac{\partial F[\mathbf{x}_\epsilon(\sigma)]}{\partial \epsilon}\right|_{\epsilon=0} =
\int\limits_{0}^{\hat{\sigma}} \nabla_{\mathbf{x}} L \cdot \mathbf{h}(\sigma)\,d \sigma +
\int\limits_{0}^{\hat{\sigma}} \nabla_{\mathbf{p}} L \cdot \mathbf{h}'(\sigma)\,d \sigma$$

We can transform the second integral using integration by parts, as follows:

$$\left. \int\limits_{0}^{\hat{\sigma}} \nabla_{\mathbf{p}} L \cdot \mathbf{h}'(\sigma)\,d \sigma =
\nabla_{\mathbf{p}} L \cdot \mathbf{h}(\sigma)\right|_{0}^{\hat{\sigma}} -
\int\limits_{0}^{\hat{\sigma}} \frac{d \nabla_{\mathbf{p}} L}{d \sigma} \cdot \mathbf{h}(\sigma)\, d\sigma\;.$$

The first term is zero because $\mathbf{h}(\sigma)$ has zero boundary values. Combining the transformed integral with the first integral in the original equation, we obtain

$$0 = 
\int\limits_{0}^{\hat{\sigma}} \left(\nabla_{\mathbf{x}} L - 
\frac{d}{d \sigma}\,\nabla_{\mathbf{p}} L \right) \cdot \mathbf{h}(\sigma)\,d \sigma\;.$$

Since the equality should hold for any $\mathbf{h}(\sigma)$, a sufficient condition for stationarity is the following differential equation, known as the *Euler-Lagrange equation*:

$$\nabla_{\mathbf{x}} L - \frac{d}{d \sigma}\,\nabla_{\mathbf{p}} L = 0\;.$$

In the mechanical analogy, the $H$ functional is analogous to energy, and the $F$ functional is analogous to action. We can take, for example,

$$L(\mathbf{x},\mathbf{p},\sigma) = \frac{S^2(\mathbf{x}) + \mathbf{p} \cdot \mathbf{p}}{2}$$

so that $\nabla_{\mathbf{x}} L = S(\mathbf{x})\,\nabla S$, $\nabla_{\mathbf{p}} L = \mathbf{p}$, and the ordinary differential equation reduces to the ray tracing equation

$$\frac{d \mathbf{p}}{d \sigma} = S(\mathbf{x})\,\nabla S$$

that we derived earlier in a different way. Note that, along a ray, $L
= S^2(\mathbf{x})$ and the stationary functional is simply the traveltime along the ray

$$F[\mathbf{x}_{\mbox{ray}}(\sigma)] = \int\limits_{0}^{\hat{\sigma}} S^2\left[\mathbf{x}_{\mbox{ray}}(\sigma)\right]\,d \sigma = 
\widehat{T}(\mathbf{x}_0,\mathbf{x}_1)\;.$$

The principle of traveltime stationarity is known as *Fermat's principle*.

Solving the eikonal equation for the minimal traveltime $T(\mathbf{x})$ on a grid is more efficiently done with finite-difference methods. The methods described in the next section compute the *viscosity solution* of the eikonal equation, which corresponds to the continuous minimum-traveltime branch of the possibly multi-valued traveltime function.
"""

# ╔═╡ bd58c3d5-ae38-4044-ba20-bce0d45a1b6e
md"""
## Finite-Difference Methods for Solving the Eikonal Equation

Designing a finite-difference method involves two choices: a finite-difference stencil and an update sequence.
"""

# ╔═╡ e209abe4-52df-4d81-9dcb-f2159db6c14a
md"""
### Finite-Difference Stencil

In this section, we will derive a first-order finite-difference stencil for the eikonal equation using two methods. Consider a 2-D finite-difference mesh where the traveltime at a current mesh point $C$ is computed from the traveltimes at its East and North neighbors, points $A$ and $B$ (see the figure below). The derivation applies to any other pair of neighbors. If the wavefront arrives from the North-East direction, the traveltime at point $C$ equals the traveltime at some point $D$ located between $A$ and $B$ plus the traveltime along the ray from $D$ to $C$.
"""

# ╔═╡ 83b4add0-65c1-4bf7-9382-f8864c4f3b11
function plot_update()
	plot([0, 0, 1, 0], [0, 1, 0, 0], color=:blue, label=:none)
	plot!([0, 1/3], [0, 2/3], linestyle=:dash, color=:blue, label=:none)
	scatter!([0, 0, 1], [0, 1, 0], label=:none, color=:blue) 
	scatter!([1/3], [2/3], label=:none, color=:white)
	annotate!([0, 0, 1, 1/3+0.01], [0, 1, 0, 2/3+0.01], 
              [L"C", L"B", L"A", L"D"], [:top, :bottom, :top, :bottom])
	plot!([0.85, 0.55], [0.25, 0.55], arrow=:closed, label=:none, color=:black)
	annotate!(0.9, 0.2, L"\xi")
	plot!(showaxis=false, grid=:none, aspect_ratio=1)
end

# ╔═╡ 0b696aeb-75b4-4ef0-a522-64fe6ae209d4
plot_update()

# ╔═╡ 7c5761a2-343a-4048-8682-8db44768de66
md"""
To derive the first-order estimate for $T_C$ (the time at point $C$), we will make two approximations:
1. 1. We will assume that the ray segment between $D$ and $C$ is a straight line. The traveltime along this straight trajectory is governed by the slowness $S_C$ at point $C$. This slowness is assumed to be locally constant.
2. We will also assume that the traveltime at point $D$ can be approximated by a linear interpolation between traveltimes at $A$ an $B$.

Both approximations are first-order accurate.
"""

# ╔═╡ 317bbbe1-0312-48be-96d6-707feafca20a
md"""
It is convenient to introduce a parameter $\xi$ along the line between $A$ and $B$ such that $A$ corresponds to $\xi=0$, $B$ to $\xi=1$, and the distance between $C$ and $D$ is $h\,\sqrt{\xi^2+(1-\xi)^2}$, where $h$ is the mesh size. For simplicity, we will assume a square mesh with equal distances between neighboring points. Thus, we arrive at the expression

$$T_C = T_A\,(1-\xi) + T_B\,\xi + S_C\,h\,\sqrt{\xi^2+(1-\xi)^2}\;,$$

where the only unknown is $\xi$. The first two terms in the equation represent the value of $T_D$ obtained by linear interpolation along the line $AB$. The third term is the constant-slowness traveltime along the ray segment $DC$. Defining $\xi$ constrains the location of $D$ and, consequently, the value of $T_C$. We can find $\xi$ by applying Fermat's principle. It is convenient to first change variables from $\xi$ to $\eta=\xi-1/2$ to make the equation more symmetric:

$$\begin{array}{rcl}
  T_C & = & T_A\,\left(\frac{1}{2}-\eta\right) + T_B\,\left(\frac{1}{2}+\eta\right)
  + S_C\,h\,\sqrt{\left(\frac{1}{2}+\eta\right)^2+\left(\frac{1}{2}-\eta\right)^2} \\
  & = & 
  T_A\,\left(\frac{1}{2}-\eta\right) + T_B\,\left(\frac{1}{2}+\eta\right)
  + S_C\,h\,\sqrt{\frac{1}{2}+2\,\eta^2}\;.
\end{array}$$

Fermat's principle states that the ray trajectory should be stationary with respect to traveltime. In this case, changing the trajectory corresponds to changing the value of $\eta$. Hence,

$$0 = {\frac{\partial T_C}{\partial \eta}} = T_B - T_A + S_C\,h\,\frac{2\,\eta}{\sqrt{\frac{1}{2}+2\,\eta^2}}\;.$$

Thus, finding $\eta$ amounts to solving a quadratic equation. The minimum-traveltime solution is

$$\eta = \frac{T_A - T_B}{2\,\sqrt{2\,S_C^2\,h^2 - (T_A - T_B)^2}}\;.$$

After substituting the equation for $\eta$ into the traveltime equation, we obtain

$$\begin{array}{rcl}
  T_C & = & \displaystyle \frac{T_A + T_B}{2} - \frac{(T_A - T_B)^2}{2\,\sqrt{2\,S_C^2\,h^2 - (T_A - T_B)^2}}
  + S_C\,h\,\sqrt{\frac{1}{2}+\frac{(T_A - T_B)^2}{2\,\left[2\,S_C^2\,h^2 - (T_A - T_B)^2\right]}} \\
\nonumber
  & = &
  \displaystyle \frac{T_A + T_B}{2} - \frac{(T_A - T_B)^2}{2\,\sqrt{2\,S_C^2\,h^2 - (T_A - T_B)^2}} +
  \frac{2\,S_C^2\,h^2}{2\,\sqrt{2\,S_C^2\,h^2 - (T_A - T_B)^2}} \\
  & = & \displaystyle \frac{T_A + T_B + \sqrt{2\,S_C^2\,h^2 - (T_A - T_B)^2}}{2}\;,
\end{array}$$

which defines the final expression for the update stencil. The derivation is analogous in the 3-D case.
"""

# ╔═╡ ce20566e-6f2f-45b9-ad5f-b302aed35beb
md"""
An alternative way to derive the update stencil follows directly from the eikonal equation

$$\nabla T \cdot \nabla T = S^2(\mathbf{x})$$

by replacing the first-order derivatives of $T$ with first-order finite-difference approximations

$$\nabla T = \displaystyle \{ \frac{\partial T}{\partial x}\,,\,\frac{\partial T}{\partial z} \} \approx
  \{ \frac{T_A - T_C}{h}\,,\,\frac{T_B - T_C}{h} \}\;.$$

The first-order finite-difference approximation of the eikonal equation is then

$$\displaystyle \left(\frac{T_C - T_A}{h}\right)^2 +  \left(\frac{T_C - T_B}{h}\right)^2 = S_C^2\;,$$

which leads to a quadratic equation for $T_C$

$$2\,T_C^2 - 2\,(T_A + T_B)\,T_C + T_A^2 + T_B^2 - S_C^2\,h^2 = 0$$

While the derivation based on direct finite-differencing of the eikonal equation may seem simpler, it helps to clarify the underlying physical assumptions. This understanding is key to both derivations and provides insights into how to increase the order of accuracy when designing finite-difference stencils for the eikonal equation.
"""

# ╔═╡ 75c2594f-ee60-4d40-9ad7-3ff84e9ed071
md"""
### Update Sequence

A finite-difference stencil tells us how to update a grid point from its neighbors. The next question is: in what order should we visit the points? This question is not evident in the eikonal equation because it lacks an obvious evolution direction (such as time in the wave equation). Several approaches exist for defining the update order.
"""

# ╔═╡ b6740101-18ac-4ba3-9bc7-3632fb51c74b
md"""
- **Paraxial method** introduces an artificial marching direction, typically depth in seismic applications, and considers only traveltimes along non-turning rays. Mathematically, this amounts to replacing the eikonal equation with the *paraxial* equation

  $${\frac{\partial
  T}{\partial z}} = \sqrt{S^2(\mathbf{y},z) - \nabla_{\mathbf{y}}
  T \cdot \nabla_{\mathbf{y}} T}\;,$$

  where $\mathbf{y}$ contains the components of $\mathbf{x}$ orthogonal to $z$, and the square root is taken to be positive so that every ray advances in the $z$ direction. The paraxial equation reduces the eikonal equation to an evolution in $z$, thereby resolving the problem of selecting the update order. In the paraxial finite-difference method, the order is defined by marching in the $z$ direction and updating values at the next $z$ level using only values from the previous level.
"""

# ╔═╡ 5690cb16-c322-482a-a82d-f034ce73c3ec
md"""
* Qian, J., and W. W. Symes, 2002, An adaptive finite-difference method for traveltimes and amplitudes: Geophysics, 67, 167–176.
"""

# ╔═╡ ddb5fe25-e8b1-46f6-a687-c74c89eefcdc
function plot_paraxial()
	plot([-1, 1], [1, 1], color=:blue, label=:none)
	plot!([0, 0], [0, 1], color=:blue, label=:none)
	plot!([-1, 0, 1], [1, 0, 1], color=:blue, label=:none)
	scatter!([-1, 0, 1, 0], [1, 0, 1, 1], label=:none, color=:blue) 
	annotate!([-1, 0, 1, 0], [1, 1, 1, 0], 
		      [L"D", L"A", L"B", L"C"], [:bottom, :bottom, :bottom, :top])
	plot!(showaxis=false, grid=:none, aspect_ratio=1)
end

# ╔═╡ 995a7f49-7f29-47d1-a176-ab18fea17dd3
plot_paraxial()

# ╔═╡ 2233c2ae-cd7a-4de7-a067-7df4a0e4b8d0
md"""
!!! assignment

    ## Task 1 (theoretical)

    Let us design a first-order finite-difference solver using the paraxial method. 

    Referring to the picture above, express the traveltime at point $C$ in terms of the traveltimes at points $A$ and $B$ and the slowness $S_C$. For simplicity, assume the mesh has equal dimensions $|AC|=|AB|=h$. How should we choose between points $A$ and $B$ versus points $A$ and $D$ for the first-order finite-difference update?
"""

# ╔═╡ 479e4cbd-696f-415b-b28d-043342a3676d
md"""
- **Fast sweeping method** is an iterative approach. One traverses the grid with iterative ``sweeps''. In the 2-D case, the sweeps can proceed from North-East, North-West, South-East, or South-West, using updates of the form derived above. Initially, the wave source points are assigned zero traveltimes, while the rest of the grid is given very large values. At each sweep, only grid points with smaller time values than the current value participate in updating their neighbors, preserving the *upwind* flow of traveltimes from the source. The word *fast* in the name of the method refers to the fact that only a limited number of sweeps is necessary to obtain a solution with first-order accuracy. The required number of iterations depends on the complexity of the velocity model.
"""

# ╔═╡ 5ec25630-1602-40ef-a22f-0c381c2ee7bf
md"""
* Tsai, Y.-H. R., L.-T. Cheng, S. Osher, and H.-K. Zhao, 2003, Fast sweeping algorithms for a class of Hamilton-Jacobi equations: SIAM Journal on Numerical Analysis, 41, 673–694.
* Zhao, H., 2005, A fast sweeping method for eikonal equations: Math. Comp., 74, 603–627.
* Fomel, S., S. Luo, and H. Zhao, 2009, Fast sweeping method for the factored eikonal equation: Journal of Computational Physics, 228, 6440–6455.
"""

# ╔═╡ 913e402c-6771-416f-b4d7-71ac26df4251
md"""
- **Adaptive nonlinear Gauss-Seidel method** uses iterations similar to the fast-sweeping method but allows more flexibility in the order of visits to grid points. The logic of the algorithm is as follows:
  1. Start by assigning traveltime values to the source points and infinite (a very large number) values to all other points on the grid.
  2. Take the neighbors of all source grid points and add them to a queue in random order.
  3. While the queue is not empty:
  * Evaluate a point at the top of the queue using its upwind neighbors and a finite-difference stencil.
  * If the new traveltime value differs from the previous value, add all neighbors of the point to the queue, skipping those already in the queue or with traveltimes smaller than the updated value.

  The iterations will converge as long as the updated values decrease, since they remain bounded below.
"""

# ╔═╡ bb3b3998-d8d0-45a9-9a49-14fc3d1d9417
md"""
* Bornemann, F., and C. Rasch, 2006, Finite-element discretization of static Hamilton-Jacobi equations based on a local variational principle: Computing and Visualization in Science, 9, 57–69.
"""

# ╔═╡ 28a36a03-7305-4ff6-887c-417646dd1b9a
md"""
- **Fast marching method** is a non-iterative approach that updates every grid point by visiting it only once. The algorithm’s logic is somewhat different:
  1. All grid points are divided into three groups: *Accepted*, *Queued*, and *Away*. Initially, the source points are *Queued* with assigned traveltime values, while the rest of the grid is assigned infinite (large) values. In the fast marching method, the queue is a fixed-grid wavefront representation.
  2. While the queue is not empty:
  * Take the *minimum traveltime* point from the queue and mark it *Accepted*.
  * Evaluate the neighbors of this point that are not *Accepted* using their upwind neighbors and a finite-difference stencil.
  * Add all previously *Away* neighbors to the queue and mark them *Queued*.  

  Unlike iterative algorithms, the fast marching method queues each grid point only once. Alternative fast-sweeping and adaptive Gauss-Seidel methods may revisit a given point multiple times during iterative updates. This advantage comes at the cost of sorting, which is required to select the minimum traveltime at each step. Efficient sorting algorithms place the *Queued* points in a heap (priority queue) structure, where extracting the minimum traveltime costs $O(\log{N})$ operations. Thus, the total operational cost of the fast marching method is $O(N\,\log{N})$. This is optimally efficient unless the number of iterations in the iterative methods can be kept smaller than $\log{N}$.
"""

# ╔═╡ 28bbf480-1dab-4047-bc2e-3c37cc38caf8
md"""
![](https://upload.wikimedia.org/wikipedia/commons/9/98/JamesSethian.jpg)
* Sethian, J., 1996, A fast marching level set method for monotonically advancing fronts: Proc. Nat. Acad. Sci., 93, 1591–1595.
* Sethian, J., 1999, Level set methods and fast marching methods: Evolving interfaces in computational geometry, fluid mechanics, computer vision and materials sciences: Cambridge University Press.
* Yatziv, L., A. Bartesaghi, and G. Sapiro, 2006, O(N) implementation of the fast marching algorithm: Journal of Computational Physics, 212, 393–399.
"""

# ╔═╡ f21bed44-22f8-4109-9fe2-9543524d5a3b
# unit velocity fo computing the distancce function
velocity = ones(1000, 1000); 

# ╔═╡ c0fd12c6-785b-45f8-ba38-2a88905c3970
npoints = 10;

# ╔═╡ 4cae06fb-5d5f-47db-8a16-d816fe5ed610
sources = rand(2, npoints)*1000 .+ 1

# ╔═╡ 1f8e7523-abee-4916-904b-04994dc6ecad
distance = FastMarching.msfm(velocity, sources, true, true);

# ╔═╡ 2da93740-a541-40ee-bdff-02ead7afbfc6
contour(distance, levels=30,
        aspect_ratio=1, c=:coolwarm, size=(600, 600), 
        title = "Distance to a set of $(npoints) source points")

# ╔═╡ 43c49069-04ff-466a-b429-7fda272ed1ca
md"""
!!! assignment

    ## Task 2

    Let us return to the rainfall example and compute the distance function on a map using data from Swiss weather stations.
"""

# ╔═╡ fad05715-f875-4a5a-a168-dc933e2ceb40
begin
	# download data files
	download("https://ahay.org/data/rain/alldata.rsf@","alldata.bin")
	download("https://ahay.org/data/rain/obsdata.rsf@","obsdata.bin")
end

# ╔═╡ 583185b9-d62d-4b1e-88ce-098752012f3e
begin
	# read data
	alldata = Array{Float32}(undef, 3, 467); 
	obsdata = Array{Float32}(undef, 3, 100); 
	read!("alldata.bin", alldata)
	read!("obsdata.bin", obsdata)
end

# ╔═╡ 4cc028a3-4f69-4934-9d72-c0df977ae794
function compute_distance(source::Array, nx::Int, ny::Int; x0=0, y0=0, dx=1, dy=1)
	# unit velocity
	velocity = ones(eltype(source), nx, ny)
	xy = similar(source[1:2,:])
	xy[1,:] = (source[1,:] .- x0)/dx .+ 1
	xy[2,:] = (source[2,:] .- y0)/dy .+ 1
	return FastMarching.msfm(velocity, xy, true, true)
end

# ╔═╡ d7579b4a-f47f-4f6c-9203-06cb68dc03e3
begin
	lat = -185:185 # latitude
	lon = -127:127 # longitude
	nx, ny = length(lat), length(lon)
	x0, y0 = lat[1], lon[1]
end

# ╔═╡ f835c777-0175-4baf-a95b-46f6e8931f62
dist = compute_distance(obsdata, nx, ny, x0=x0, y0=y0);

# ╔═╡ eade099c-cb63-4b10-9592-25af1b8701e8
heatmap(lat, lon, dist', title="Distance", cmap=:coolwarm)

# ╔═╡ 3eb45455-e7c1-4343-8d32-25c871e1ec3b
md"""
Once we have the distance function for different grid points, we can proceed to assign rainfall values to these points based on the nearest weather station.
"""

# ╔═╡ 733c258e-c0c7-41b4-bc77-48b36d8e2012
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

# ╔═╡ bf10cd45-8e28-43b2-8ecc-0c285b709eaf
binned = bin(obsdata, obsdata[3,:], nx, ny, x0=x0, y0=y0);

# ╔═╡ 5e766c4b-b33e-428d-a1ab-235928a32adf
CartesianIndices(dist)[:]

# ╔═╡ 3f9ca063-e2a6-4862-b166-e0832eefcffb
function nearest_neighbor(distance)
	# sort grid indices by distance
	grid = CartesianIndices(distance)[:]
	sort!(grid, by=i -> distance[i])
	neighbor = similar(grid)
	d1, d2 = CartesianIndex((1,0)), CartesianIndex((0,1))
	@inbounds for i in 1:length(grid)
		g = grid[i]
    	neighbors = filter(x -> x in grid, [g, g+d1, g-d1, g+d2, g-d2])
    	d, j = findmin(distance[neighbors])
    	neighbor[i] = neighbors[j]
	end
	return hcat(grid, neighbor)
end

# ╔═╡ c8393b67-e0e4-414f-bb52-864082acf11e
grid = nearest_neighbor(dist);

# ╔═╡ 369cea5f-c3bf-47ff-af73-d01489d8b15b
function interpolate_nearest(binned, grid)
	nearest = copy(binned)
	for (i, j) in eachrow(grid)
    	nearest[i] = nearest[j]
	end
	return nearest
end

# ╔═╡ 9c3c01f8-8ae2-4490-9a11-44f37b14c505
rain = interpolate_nearest(binned, grid);

# ╔═╡ 10fd537a-1505-4827-8904-8008c1dd3ecd
heatmap(lat, lon, rain', title="Rainfall Interpolation", cmap=:viridis)

# ╔═╡ 7d733b2f-1c90-4b0d-83fa-43de9e31bbcf
md"""
**Your task**: compare the computational cost of `interpolate_nearest` with that of triangulated linear interpolation, both theoretically and empirically.
"""

# ╔═╡ 47b2ea6f-02c3-4a43-b032-f8899626ae30
md"""
!!! assignment

    ## Task3
	Return to the GMRES-based shaping regularization method from the previous assignment and replace triangulated linear interpolation with nearest-neighbor interpolation. Choose appropriate parameters and compare the accuracy of the results.
"""

# ╔═╡ 40a37228-1e59-413a-8eae-55fd2b16f5af
backward2(v) = interpolate_nearest(bin(obsdata, v, nx, ny, x0=x0, y0=y0), grid);

# ╔═╡ 23fbbe58-ed3d-46fe-9579-b7d560c4ba09
md"""
!!! assignment

    ## Bonus Task

    For extra credit, analyze a different dataset. The following dataset comes from the 2004 Spatial Interpolation Comparison, a contest organized by the Radioactivity Environmental Monitoring Group (Institute for Environment and Sustainability, Joint Research Centre, European Commission). The data are daily dose-rate measurements reported by the German National Automatic Monitoring Network (IMIS).

	*  Dubois G, Galmarini S.: Introduction to the Spatial Interpolation Comparison (SIC) 2004 Exercise and Presentation of the Datasets, Applied GIS, Vol. 1, No. 2, 2005, ISSN 1832-5505 (DOI: 10.2104/ag050009).
"""

# ╔═╡ 30c9b705-b838-4f6d-9677-e98c07517b6f
begin
	download("https://ahay.org/data/geostat/SIC2004_input.csv","input.csv")
	download("https://ahay.org/data/geostat/SIC2004_out.csv","output.csv")
end

# ╔═╡ ed358754-1d28-4719-8e6d-90003c3eaa25
input = DataFrame(CSV.File("input.csv", header=["station", "x", "y", "value"]))

# ╔═╡ 5cb93001-478f-40f2-98d1-701dd562319f
scatter(input.x/1000, input.y/1000, label=:none, size=(400, 800), aspect_ratio=1,
	    xlabel="X (km)", ylabel="Y (km)", title="Input Station Locations")

# ╔═╡ d0eef3bc-541f-436c-a496-371697dd4779
md"""
**Your Task**: Try several different interpolation methods and compare their accuracy.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
FastMarching = "7c16e180-9f04-11e8-24a6-e7c7f74617b0"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
CSV = "~0.10.15"
DataFrames = "~1.7.1"
FastMarching = "~0.2.7"
HTTP = "~1.10.15"
LaTeXStrings = "~1.4.0"
Plots = "~1.40.9"
PlutoUI = "~0.7.79"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "6acf5c8e4af2753251e1fe8de8811ea85b367ae0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

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

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

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

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "f36e5e8fdffcb5646ea5da81495a5a7566005127"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.3"

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

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

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

[[deps.FastMarching]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "58174c1474002bd7ed48771b841b635b4ce97373"
uuid = "7c16e180-9f04-11e8-24a6-e7c7f74617b0"
version = "0.2.7"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

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

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

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

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a3c1536470bf8c5e02096ad4853606d7c8f62721"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.2"

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
# ╟─f4712ba3-c259-4782-8656-ec8ee53d701c
# ╟─8bd6f886-ffca-417e-99cd-ac98f5a0460b
# ╟─c6235c5d-ecca-421f-bbcf-65af557b9758
# ╟─02882467-033a-42a7-97e6-02d8b6e309b7
# ╟─de2cae4e-d7bf-4484-96d0-4e3ffa5afd0b
# ╟─0f1cb87f-60f4-464b-9ead-71940e81132b
# ╟─0a821bd1-80e0-4c16-8e71-f8eb46cd9563
# ╟─95f77583-64c6-4864-a1db-da6cd99c13ee
# ╟─16b57010-35b7-4353-90e4-2d523c794e60
# ╟─bd58c3d5-ae38-4044-ba20-bce0d45a1b6e
# ╟─e209abe4-52df-4d81-9dcb-f2159db6c14a
# ╠═01ba7a19-4f88-472a-9679-39af1a109b0d
# ╠═280f7209-6291-43aa-926e-918497cf9d24
# ╟─83b4add0-65c1-4bf7-9382-f8864c4f3b11
# ╠═0b696aeb-75b4-4ef0-a522-64fe6ae209d4
# ╟─7c5761a2-343a-4048-8682-8db44768de66
# ╟─317bbbe1-0312-48be-96d6-707feafca20a
# ╟─ce20566e-6f2f-45b9-ad5f-b302aed35beb
# ╟─75c2594f-ee60-4d40-9ad7-3ff84e9ed071
# ╟─b6740101-18ac-4ba3-9bc7-3632fb51c74b
# ╟─5690cb16-c322-482a-a82d-f034ce73c3ec
# ╟─ddb5fe25-e8b1-46f6-a687-c74c89eefcdc
# ╠═995a7f49-7f29-47d1-a176-ab18fea17dd3
# ╟─2233c2ae-cd7a-4de7-a067-7df4a0e4b8d0
# ╟─479e4cbd-696f-415b-b28d-043342a3676d
# ╟─5ec25630-1602-40ef-a22f-0c381c2ee7bf
# ╟─913e402c-6771-416f-b4d7-71ac26df4251
# ╟─bb3b3998-d8d0-45a9-9a49-14fc3d1d9417
# ╟─28a36a03-7305-4ff6-887c-417646dd1b9a
# ╟─28bbf480-1dab-4047-bc2e-3c37cc38caf8
# ╠═f201996f-84a7-4f7d-ab32-6647515025f7
# ╠═f21bed44-22f8-4109-9fe2-9543524d5a3b
# ╠═c0fd12c6-785b-45f8-ba38-2a88905c3970
# ╠═4cae06fb-5d5f-47db-8a16-d816fe5ed610
# ╠═1f8e7523-abee-4916-904b-04994dc6ecad
# ╠═2da93740-a541-40ee-bdff-02ead7afbfc6
# ╟─43c49069-04ff-466a-b429-7fda272ed1ca
# ╠═fad05715-f875-4a5a-a168-dc933e2ceb40
# ╠═583185b9-d62d-4b1e-88ce-098752012f3e
# ╠═4cc028a3-4f69-4934-9d72-c0df977ae794
# ╠═d7579b4a-f47f-4f6c-9203-06cb68dc03e3
# ╠═f835c777-0175-4baf-a95b-46f6e8931f62
# ╠═eade099c-cb63-4b10-9592-25af1b8701e8
# ╟─3eb45455-e7c1-4343-8d32-25c871e1ec3b
# ╠═733c258e-c0c7-41b4-bc77-48b36d8e2012
# ╠═bf10cd45-8e28-43b2-8ecc-0c285b709eaf
# ╠═5e766c4b-b33e-428d-a1ab-235928a32adf
# ╠═3f9ca063-e2a6-4862-b166-e0832eefcffb
# ╠═c8393b67-e0e4-414f-bb52-864082acf11e
# ╠═369cea5f-c3bf-47ff-af73-d01489d8b15b
# ╠═9c3c01f8-8ae2-4490-9a11-44f37b14c505
# ╠═10fd537a-1505-4827-8904-8008c1dd3ecd
# ╟─7d733b2f-1c90-4b0d-83fa-43de9e31bbcf
# ╟─47b2ea6f-02c3-4a43-b032-f8899626ae30
# ╠═40a37228-1e59-413a-8eae-55fd2b16f5af
# ╟─23fbbe58-ed3d-46fe-9579-b7d560c4ba09
# ╠═30c9b705-b838-4f6d-9677-e98c07517b6f
# ╠═b99b18c8-bfb5-46f9-b733-1ff817d1f1ae
# ╠═ed358754-1d28-4719-8e6d-90003c3eaa25
# ╠═5cb93001-478f-40f2-98d1-701dd562319f
# ╟─d0eef3bc-541f-436c-a496-371697dd4779
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
