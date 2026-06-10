### A Pluto.jl notebook ###
# v0.20.27

#> [frontmatter]
#> chapter = 4
#> section = 3
#> order = 3
#> image = "https://raw.githubusercontent.com/fonsp/Pluto.jl/580ab811f13d565cc81ebfa70ed36c84b125f55d/demo/plutodemo.gif"
#> title = "Second Data Analysis Notebook"
#> tags = ["module4", "track_julia", "track_material", "Pluto", "PlutoUI"]
#> layout = "layout.jlhtml"

using Markdown
using InteractiveUtils

# ╔═╡ 097057c3-8fa7-4c78-b44d-4536c6129cd4
begin
	using PlutoUIExtra
	TableOfContents()
end

# ╔═╡ fb214786-5adf-4709-a6d6-48923e81b3bb
using Plots

# ╔═╡ 5d4561d7-4dec-46c1-ab23-8e0084023e6c
using Random

# ╔═╡ 89578615-9355-4193-b11a-c2456a20a184
using BenchmarkTools # for benchmarking

# ╔═╡ da3878f5-6d52-4228-a076-4e27118cac01
using PyCall

# ╔═╡ 43447d2b-c1aa-4dae-b98d-df3ce412496e
using DataFrames

# ╔═╡ 45a88f15-932d-45ac-a12b-a99a880c1f26
using LaTeXStrings

# ╔═╡ 0be18d16-10f0-4e8b-9876-3ee5d95cb4d3
using CSV

# ╔═╡ 6166a05a-310b-4b1d-95b3-275cea7cec94
import HTTP

# ╔═╡ a43bef1c-c149-4d4e-8ff3-15e07dad1cf8
let
    notebook_dir = @__DIR__
    toc_path = joinpath(notebook_dir, "TOC_Notebook.jl")
    
    Markdown.parse("[⬅️ Back to Table of Contents](./open?path=$(HTTP.escapeuri(toc_path)))")
end

# ╔═╡ b466ed32-1534-423b-9cdf-cdd71f3da08a
md"""
# Averaging

Fundamentally, digital data are sequences of numbers. This chapter covers various methods for summarizing a sequence of numbers into a single value — a global data attribute representing the average.
"""

# ╔═╡ e79cd5b6-edae-4238-92a8-c4ddb5a2d44d
md"""
Let the data vector be a sequence of $N$ numbers $a_1, a_2, \ldots, a_N$. The averaging attribute replaces this sequence with an average value $\alpha$. There are several ways to define the average. I will introduce some definitions first and explore their deeper meaning later.
"""

# ╔═╡ 13f1c675-3138-414b-827e-a0c2043cca73
md"""
## Mean

The mean is defined as

$\alpha = \frac{1}{N} \sum\limits_{n=1}^{N} a_n\;.$
"""

# ╔═╡ 03e24dff-1b3f-4651-91aa-a531b14931f4
md"""
The calculation of the mean is fundamentally a linear operation. It can be expressed as multiplying by a row vector, as follows:

$\alpha = \left[\begin{array}{cccc} \frac{1}{N} & \frac{1}{N} & \cdots &  \frac{1}{N}\end{array}\right]\,\left[\begin{array}{c} a_1 \\ a_2 \\ \vdots \\ a_N\end{array}\right]\;.$
"""

# ╔═╡ 9a711e15-0f99-4c83-8472-b5767508276a
mean(vector) = sum(vector)/length(vector)

# ╔═╡ b2d1e843-1efd-49fc-b30c-62d86093c2a4
mean([2 0 2 6])

# ╔═╡ f04f5adc-c0d7-4b1c-afc2-342815d9c529
md"""
If a random number $x$ exists within the range $a \le x \le b$ and has a probability density function $f(x)$, the *mathematical expectation* of this number is defined as

$$E[x] = \int\limits_a^b x\,f(x)\,dx\;.$$

In other words, the mathematical expectation is the centroid (center of gravity) of the area under the probability density curve. 
"""

# ╔═╡ a66b0645-ca58-48e9-8442-1fc2fe17bca0
md"""
If we assume that all data values $a_1, a_2, \ldots, a_N$ are random and identically distributed with the mathematical expectation $a$, then the mathematical expectation of $\alpha$ is

$$E[\alpha] = \frac{1}{N} \sum\limits_{n=1}^{N} E[a_n] = a\;.$$
"""

# ╔═╡ 7c0f5540-b9ed-4d73-af37-462d7a245acc
md"""
Moreover, if the variance of each data value $a_n$ is $E[(a_n-a)^2]=\sigma^2$ and the data values are statistically independent, then the variance of the mean is
"""

# ╔═╡ f375d3af-81a8-4636-943a-a341062139b8
md"""
$$\begin{array}{rcl}E[(\alpha-a)^2] & = & \displaystyle E\left[\left(\frac{1}{N} \sum\limits_{n=1}^{N} a_n - a\right)^2\right] \\ & = & \displaystyle 
  \frac{1}{N^2} \sum\limits_{n=1}^{N} E\left[(a_n - a)^2\right] = \frac{\sigma^2}{N}\end{array}$$

and is approaching zero with increasing $N$. 
"""

# ╔═╡ 1453c4e4-540a-40e6-9893-60051b89b841
md"""
This shows that the mean offers a stable and unbiased estimate of the expected value.
"""

# ╔═╡ b80342c8-c673-4215-ad1a-b03cdd205e4a
md"""
## Median

The mean as an average makes intuitive sense for numbers within narrow probabilistic ranges. For example, calculating the average height of a group of people is meaningful because human height is generally limited and tends to cluster around an average. Conversely, using the mean income of a group can be highly misleading (if Elon Musk suddenly enters the room).  
"""

# ╔═╡ a7898225-7f51-410a-8b70-b03288792683
md"""
For unbalanced distributions, a more suitable measure of the average is the median, defined as the middle value of the data set when it is sorted by value.  *median*, which is defined as tIf the original sequence is $a_1, a_2, \ldots, a_N$ and the sorted sequence is $a_{k_1}, a_{k_2}, \ldots, a_{k_N}$, then the mean is defined as $a_{k_{(N+1)/2}}$ if $N$ is odd and $\left(a_{k_{N/2}}+a_{k_{N/2+1}}\right)/2$ if $N$ is even. In other words, it is a number greater than or equal to half of the data values and less than or equal to numbers from the other half.
"""

# ╔═╡ 63dc5fd8-51ae-44ab-b938-eef4462e1d80
md"""
For an illustration, let us look at two different distributions of random numbers.
"""

# ╔═╡ 68904d0b-4cc7-4951-b3b1-d146ecfc93c9
a = exp.(randn(10000)*0.05)

# ╔═╡ 21edb74e-6434-4b45-b0ac-8755e90f5992
import Statistics

# ╔═╡ ae9c3b2e-af40-4ad4-96a9-93490420842e
a_mean, a_median = mean(a), Statistics.median(a)

# ╔═╡ 23774536-d1f0-4f0e-ae46-cc15a5a6fa3c
begin
	a_plot = plot(a, title="Random Sequence", label=:none, ylabel="Amplitude");
	a_hist = stephist(a, title="Histogram", xlabel="Amplitude", label=:none);
	plot!(a_hist, [a_mean], seriestype="vline", label="mean");
	plot!(a_hist, [a_median], seriestype="vline", label="median");
	plot(a_plot, a_hist, layout=(2,1))
end

# ╔═╡ 5eb96318-b142-4479-a6c8-d06a44a75087
a2 = exp.(randn(10000)*0.5);

# ╔═╡ 8fff5fd1-33d3-46a6-afe5-ed9a6ade5c67
a2_mean, a2_median = mean(a2), Statistics.median(a2)

# ╔═╡ 7acffe75-e56a-44f7-821a-b3641f90022a
begin
	a2_plot = plot(a2, title="Random Sequence", label=:none, ylabel="Amplitude");
	a2_hist = stephist(a2, title="Histogram", xlabel="Amplitude", label=:none);
	plot!([a2_mean], seriestype="vline", label="mean");
	plot!([a2_median], seriestype="vline", label="median");
	plot(a2_plot, a2_hist, layout=(2,1))
end

# ╔═╡ cb379364-095f-4ab0-9609-425313be3fa0
md"""
We can observe that the mean and median may differ when the distribution is unbalanced.
"""

# ╔═╡ c2225da8-7903-459f-8722-280475dbdc1d
md"""
## Quantile

A generalization of the median is *quantile*. When the data values are sorted from small to large, the $p\%$ quantile is the value $a_{k_n}$, which appears at $n=N\,\frac{p}{100}$ place in the sequence. With that definition, the median corresponds to the $50\%$ quantile.
"""

# ╔═╡ e0a3bd12-0fac-484a-986f-feb514d763d3
md"""
### Sorting algorithms

Finding a quantile value requires sorting the input data sequence. Let's examine various methods for sorting.
"""

# ╔═╡ f95a8a8d-ee00-4730-9959-75f2569563fe
md"""
#### Insertion sort

One of the simplest ways to sort a list of numbers is to add them to an initially empty list one at a time, comparing each new number with the existing numbers in the sorted list and shifting the list if necessary. This basic sorting method is similar to how most people would organize books on a shelf alphabetically. The following code demonstrates this method:
"""

# ╔═╡ 8bf5b38f-2fe3-4c14-9c8b-677195a1a726
function slowsort(list)
    n = length(list) 
	list2 = copy(list)
    for k in 1:n # 1,2,...,n
        # everything up to k-1 is sorted
        for j in k:-1:2 # k,k-1,...,2
            item, prev = list2[j], list2[j-1]
            if item > prev; break; end # out of the loop  
            list2[j-1], list2[j] = item, prev
        end
    end
	return list2
end

# ╔═╡ cd713688-43e0-4e08-a2a4-b80c0c66ea9b
begin
	test = collect(1:100)
	shuffle!(test)
end

# ╔═╡ 9c6c910e-65ec-4ec5-8a7c-35289c8e70d7
slowsort(test)

# ╔═╡ 16639520-f0c2-4518-93d7-934f69615232
md"""
How many operations does insertion sort require? 

Because of the two nested loops involved, both the worst-case scenario (trying to sort a sequence in reverse order) and the average-case scenario (sorting a random sequence) require $O(N^2)$ comparison operations. There is a way to make sorting much faster - the *quicksort* algorithm.
"""

# ╔═╡ a67e9dca-573c-4d81-a663-4661afe7c4c1
md"""
#### Quicksort

The main idea behind quick sorting is the observation that the task of sorting can be divided into two parts. First, select a number from the list, then split the list into two: one with numbers smaller than the chosen number and the other with numbers greater than it. Next, sort each of the two sublists using the same method until they become short enough (one or zero numbers) that no further sorting is needed.   
"""

# ╔═╡ 1555a6bc-eaf2-47a3-ae90-c59c656fdcda
md"""
The strategy of breaking a task into smaller parts repeatedly is known as *divide and conquer* and forms the basis for many efficient algorithms. Below is a simple code example of divide-and-conquer sorting.
"""

# ╔═╡ f1c6c404-3bbf-4c20-b40d-e821c01d914c
function quicksort(list::Array)
    if length(list) <= 1; return list; end # no sorting
    pivot = list[1] 
    Type = typeof(pivot)
	# create empty lists
    less, equal, more = Type[], Type[], Type[]  
    for item in list
        if item < pivot
            push!(less, item)
        elseif item > pivot
            push!(more, item)
        else
            push!(equal, item)
        end
    end
    # concatenate three lists
    return vcat(quicksort(less), equal, quicksort(more))
end

# ╔═╡ 19d39c97-c18b-432f-a7b4-0ecf2c90a247
quicksort(test)

# ╔═╡ ed06a444-82bd-4d15-82f5-639570e2fce7
md"""
How many operations does the quick sort take? According to the divide and conquer strategy, the cost $C(N)$ for sorting a sequence of $N$ numbers corresponds to twice the cost of sorting half of the numbers plus the linear cost of dividing the sequence into two lists. 
"""

# ╔═╡ e3819bf5-3976-4cf0-a8bc-0477950f35ed
md"""
Thus, $C(N)$ obeys the recursion

$$C(N) = 2\,C(N/2)+O(N)=2\,C(N/2)+\gamma\,N$$

with a constant $\gamma$. The big-O notation $O(N)$ here indicates a
cost which is growing proportionally to $N$ as $N$ becomes large. The
recursion starts with the zero cost for sorting one number:
$C(1)=0$. 
"""

# ╔═╡ 254daf4a-73d4-4d8d-974a-5fbf467f904f
md"""
Let us define the function $R(N)=C(N)/N$. Dividing the previous equation by $N$, we arrive at the recursion

$$R(N) = R(N/2)+\gamma$$

with $R(1)=0$. It is easy to see that $R(N)=\gamma \log_2 N$ and
therefore $C(N)=O(N\,\log N)$. 
"""

# ╔═╡ f728dd3b-732e-4ea4-8848-a5177172cf37
md"""
| $N$ | $O(N)$ | ${O(N{\log}N)}$ | $O(N^2)$ | $O(N^3)$ |
|:----|:-------|:----------------|:---------|:--------|
| $100$   | 1 sec | 7 sec  | 2 min    | 3 hours |
| $1000$   | 1 sec | 10 sec | 17 min   | 12 days |
| $1000^2$ | 1 sec | 20 sec | 12 days  | 32 thousand years |
| $1000^3$ | 1 sec | 30 sec | 32 years | 32 billion years |
"""

# ╔═╡ aa2cd69d-816b-4fa4-8dfe-22a946cb7c5c
md"""
How much faster is $O(N\log N)$ compared to $O(N^2)$? The table above shows a simple comparison of run times for algorithms with different computational complexities. If an $O(N)$ algorithm takes 1 second to compute with $N=1,000^3$ (a modest-size data cube), the corresponding $O(N^2)$ algorithm will take about 32 years. The $O(N^3)$ algorithm, on the other hand, would take over 32 billion years, far longer than the age of the known universe.  
"""

# ╔═╡ e8d7d3b0-2b16-4f5c-a3ad-4a30261cc58d
md"""
!!! note

    When working with large-scale data of size $N$, we can only afford algorithms with a computational complexity of $O(N)$ or $O(N\log N)$.
"""

# ╔═╡ 90ee6345-bd84-453f-91b5-1a0a8a878d93
md"""
### Comparing computational time

Let us measure the running time of different functions experimentally. We can measure the computational time using the `@time` macro in Julia.
"""

# ╔═╡ 5bfd4ea4-a67d-4643-9c5e-ce1cf62049f8
x = rand(10^3); # random array

# ╔═╡ f90209c1-e32c-4c00-8ce0-8d33c4904f0d
@time quicksort(x)

# ╔═╡ 46b2cd69-a94f-4f5d-8ada-b9348c946dc5
@time slowsort(x)

# ╔═╡ 9b206b00-0f15-415b-a4a7-f9b8a6c8cad1
md"""
For small sizes of the input data (1,000 in this example), `slowsort()` can be competitive with `quicksort()`. But what happens if we increase the size?
"""

# ╔═╡ e2242cbc-752c-4b34-8f6d-4f807ccef8b9
x2 = rand(10^5); # larger size

# ╔═╡ bb83acd2-27a7-4ef1-b373-3b3d0f341bc8
@time quicksort(x2)

# ╔═╡ 759b2a47-ff95-4e73-99fb-c26497b9dcf6
@time slowsort(x2)

# ╔═╡ d74bbfaf-2b70-4c44-b64a-51ec771a14da
md"""
A more accurate benchmarking comparison of the running time is provided by Julia's `BenchmarkTools` package.
"""

# ╔═╡ c885494b-e081-4042-bc99-cb76c9bffea3
@btime slowsort($x2);

# ╔═╡ 43b357f7-d48e-4641-999c-8913358f7527
@btime quicksort($x2);

# ╔═╡ 9b311d95-ecf6-48d7-bf66-da06f6c3ee4d
# Julia standard library function
@btime sort($x2, alg=QuickSort);

# ╔═╡ f00594ad-2eeb-4f76-9797-fe6b2da32895
md"""
How does the speed of Julia's code compare to Python's? It's easy to check since we can call Python functions directly from Julia.
"""

# ╔═╡ 15fd0528-6304-4d0c-8ef1-41a451699b24
# Python standard library function
pysort = pybuiltin("sorted")

# ╔═╡ 831bdedf-72f5-4317-a56b-f167e850e991
pysort(test)

# ╔═╡ bdacf5de-2ce5-4bea-94a1-41a75df0b16d
@btime pysort($x2);

# ╔═╡ 92d12878-d2df-4e1b-868f-f8979b1bcd58
np = pyimport("numpy")

# ╔═╡ 0352b81a-7f0b-49db-9e7c-f2cf69fa2bab
np.sort(test)

# ╔═╡ cc8fbae9-c9d2-41e5-87a0-529fca6d9292
@btime np.sort($x2);

# ╔═╡ 57363200-b2f5-46af-b33d-ca2bd06d0462
begin
	# Writing our own quicksort in Python
	
	py"""
	def quicksort(list):
	    if len(list) <= 1:
	        return list
	    less, equal, more = [], [], [] # empty lists
	    pivot = list[0]
	    for item in list:
	        if item < pivot:
	            less.append(item)
	        elif item > pivot:
	            more.append(item)
	        else:
	            equal.append(item) 
	    return quicksort(less) + equal + quicksort(more)
	"""
	py_quicksort = py"quicksort"
end

# ╔═╡ 63463017-11a6-4e17-a3fb-71b47fb6ff7d
py_quicksort(test)

# ╔═╡ c042c4a5-f03e-425d-b195-5918d2126dfb
@btime py_quicksort($x2);

# ╔═╡ 80353213-14cc-423f-accd-97efe9a1d0ac
# dictionary of functions
sorting = Dict(
    "Slow (own verion)" => slowsort,
    "Quick (own verion)" => quicksort,
    "Julia standard library" => sort,
    "Python standard library" => pysort,
    "Python NumPy library" => np.sort,
    "Python (own version)" => py_quicksort
    )

# ╔═╡ 584ee3fc-2e04-4ab4-899d-4ddcf7289876
begin
	df = DataFrame(Implementation = String[], Time = Float64[])
	for (name, func) in sorting
	    # @belapsed extracts the runtime in seconds
	    time = @belapsed $func($x2);
	    push!(df, [name, time])
	end
	sort!(df, :Time)
end

# ╔═╡ 85303736-a9ef-43fc-b1c1-d7babb04aec0
# extended benchmarking information
@benchmark quicksort($x2)

# ╔═╡ 8ff320e7-2637-4c32-bfdc-1f014a7cbd94
md"""
!!! assignment
    ## Task 1

    To improve the performance of the insertion sort algorithm, notice that it does one unnecessary assignment inside the inner loop. Removing this assignment leads to the following slightly longer code:
"""

# ╔═╡ 42d340a6-8566-4b1f-849e-ce7243da8d64
function slowsort_new(list)
    n = length(list) 
	list2 = copy(list)
    for k in 1:n 
        # everything up to k-1 is sorted
        item = list2[k]
        for j in k-1:-1:1 # k-1,k-2,...,1
            prev = list2[j]
            if item > prev; break; end # out of the loop  
            list2[j+1], list2[j] = XXXX, YYYY # FIX ME!
        end
    end
	return list2
end

# ╔═╡ 6d72741b-a9a6-4dc7-9341-647e56ce0b16
md"""
Fix the code above by replacing `XXXX` and `YYYY` with the correct variable names. You can then test it by sorting test lists of integers and floating-point numbers.
"""

# ╔═╡ 83f8b722-c7bb-452f-9290-3850daf62cb9
## Uncomment below to test slowsort_new
# slowsort_new(test)

# ╔═╡ 21f33ba9-d33c-4122-b47d-9cd30795c57b
## Uncomment below to test slowsort_new
# slowsort_new(x)

# ╔═╡ e104818b-4d93-44b9-8437-81a3376b7f3c
md"""
Finally, use `@btime` to benchmark the computational time of `slowsort_new` and compare it with `slowsort`.
"""

# ╔═╡ 75a30393-5683-4c7f-96dd-1f9aec648e0c


# ╔═╡ d94dcc77-3626-4625-80c7-869cd49d5554
md"""
!!! assignment
    ## Task 2

    Our implementation of the quick sort algorithm suffers from the need to allocate and deallocate many arrays, which slows down its performance. Below is a version of the code that does sorting in place without allocating new storage. It follows the same logic of dividing the list into two parts but creates those parts by shifting the items in the original list.
"""

# ╔═╡ 82bfbdc9-5a74-42b2-99c6-d41cbb78fdf8
function quicksort!(list, left=1, right=length(list))
    if (left >= right); return; end # no need to sort
    pivot = list[left]
    l, r = left, right
    # separate into two lists
    while l < r
        item = list[l+1]
        if item <= pivot
            l += 1
        else 
            list[l+1] = list[r]
            list[r] = item
            r -= 1
        end
    end
    list[left], list[l] = list[l], pivot
    # sort each list
    quicksort!(list, left, l-1)
    quicksort!(list, l+1, ZZZZ) # FIX ME!
end

# ╔═╡ 7bbe4f4d-cc17-41d0-a167-f96eb8a23015
md"""
Fix the code by replacing `ZZZZ` with the correct variable name. You can then test it by sorting test lists of integers and floating-point numbers.
"""

# ╔═╡ 6491eaf2-de10-4336-9ed1-5969836abdf8
## Uncomment below to test quicksort!
# begin
#    test1 = copy(test)
#    quicksort!(test1)
#    test1
# end

# ╔═╡ 646d7090-0594-424b-82d1-f6a43433976d
## Uncomment below to test quicksort!
# begin
#    x1 = copy(x)
#    quicksort!(x1)
#    x1
# end

# ╔═╡ a7e8f665-78bf-4338-8ef8-5f69d59bce60
md"""
Finally, use @btime to benchmark the computational time of `quicksort!` and compare it with `quicksort`. Note that `quicksort!` modifies the input by sorting it in place, while `quicksort` returns a new list.
"""

# ╔═╡ 82ec41fa-9a8f-451c-8a48-de688c8833eb
# @btime quicksort!($x1) setup=(shuffle!($x1))

# ╔═╡ a5c5139c-d0b5-4086-8930-55d41c23f064
md"""
!!! assignment
    ## Task 3

    We can use a simple animation to illustrate the behavior of the two algorithms (the insertion sort and the quick sort.)
"""

# ╔═╡ d1934ea3-d942-48b3-8197-3abc11c9afca
function plot_slowsort(list::Array)
    n = length(list) 
	list2 = copy(list)
    anim = @animate for k in 1:n 
        scatter(list2, legend=false, title="Slow Sort")
        # everything up to k-1 is sorted
        for j in k:-1:2 # k,k-1,...,2
            item, prev = list2[j], list2[j-1]
            if item > prev; break; end # out of the loop  
            list2[j-1], list2[j] = item, prev
        end
    end
    return anim
end

# ╔═╡ fafba457-1edf-4064-a895-83c80f85b168
anim = plot_slowsort(test);

# ╔═╡ 3a8c90c5-fe07-4b0a-b33b-46e5443f47f8
gif(anim, "slowsort.gif", fps = 10)

# ╔═╡ 79913fef-f496-4b06-97cc-d063ee5e6f67
function plot_quicksort!(anim, list, left=1, 
                         right=length(list))
    if (left >= right); return; end # no sorting
    plt = scatter(list, color=:green,
                  legend=false, title="Quick Sort")
    frame(anim, plt) # add frame to animation
    pivot = list[left]
    l, r = left, right
    # separate into two lists
    while l < r
        item = list[l+1]
        if item <= pivot
            l += 1
        else 
            list[l+1] = list[r]
            list[r] = item
            r -= 1
        end
    end
    list[left], list[l] = list[l], pivot
    # sort each list
    plot_quicksort!(anim, list, left, l-1)
    plot_quicksort!(anim, list, l+1, ZZZZ)
end

# ╔═╡ 6d096f2c-5876-4465-a257-219b7c825842
md"""
Replace `ZZZZ` in the code above with the correct variable name, uncomment the code below, and add one line to create an animated GIF file called "quicksort.gif" with an animation rate of 10 frames per second.
"""

# ╔═╡ 01b4fc8c-c48c-4e90-b65c-c578170c282d
## !!! Uncoment below
# begin
#    anim2 = Animation()
#    atest = copy(test)
#    plot_quicksort!(anim2, atest)
## !!! Add one line
# end

# ╔═╡ 24a2397d-d209-408b-acfd-1355afda0c23
md"""
### Quick quantile

When calculating a quantile value, such as the median, it is not necessary to fully sort the list. Instead, we can partition the list into two parts, similar to quicksort, and then select the quantile from only one of the two sublists. The code below implements this method.
"""

# ╔═╡ ee3c1632-1939-4b27-820a-f9f35d2f7196
function quantile(list, n::Int)
    pivot = list[1] 
    n_equal = 0 # number of items = pivot
    Type = typeof(pivot)
    less, more = Type[], Type[] # empty lists
    for item in list
        if item < pivot
            push!(less, item)
        elseif item > pivot
            push!(more, item)
        else
            n_equal += 1
        end
    end
    n_less = length(less)
    if n <= n_less; return quantile(less, n); end
    n_less += n_equal
    if n <= n_less; return pivot; end
    return quantile(more, n-n_less)
end

# ╔═╡ c8101efa-eee2-4a20-b8c4-18c2ad6d44a2
test

# ╔═╡ 677b4943-c5ca-4253-9744-0957d4ded983
quantile(test, 42)

# ╔═╡ 159b140f-aee2-4d49-aa9c-60f679f448b4
for i=1:100
    @assert(quantile(test, i) == i)
end

# ╔═╡ 80a13b89-0729-40c9-8368-cb5afa35394b
md"""
The cost of the quick quantile algorithm satisfies the relationship

$$C(N) = C(N/2)+O(N)=C(N/2)+\gamma\,N$$

with $C(1)=0$. It follows that, in this case, $C(N)=2\,\gamma\,(N-1)=O(N)$.
"""

# ╔═╡ da2a9350-2a28-4fc3-84a6-76241a36f045
function median(list)
    n = length(list)
    if n % 2 == 0 # n is even
        # n÷2 is integer division: same as div(n,2)
        return (quantile(list, n÷2) + 
                quantile(list, n÷2 + 1))/2
    else # n is odd
        return quantile(list, (n + 1)÷2)
    end
end

# ╔═╡ fe8ef499-7bd8-443f-a303-f403ae4b9116
median(test)

# ╔═╡ 9c057265-c7f8-40b1-bf43-87ad3fce5ff3
@btime median(x2)

# ╔═╡ ef73ca95-7c99-4fe2-9cba-4cb4cc0a8459
@btime Statistics.median(x2)

# ╔═╡ e93be60d-5a9c-4dee-9155-c9be61c29fc2
md"""
!!! assignment
    ## Task 4

    Our experiment should involve different data sizes to compare the efficiency of various algorithms systematically.
"""

# ╔═╡ da1a4879-61cd-4263-b675-89b4c07aa91a
function benchmark(sizes::Array{Int})
   na = length(sizes)
   arrays = rand.(sizes)
   # create a data frame
   bm = DataFrame(Sizes = sizes)
   # dictionary of algorithms to compare
   algorithms= Dict("slowsort" => slowsort, 
                    "quantile" => median)
   for (name, func) in algorithms
      times = Array{Float64}(undef,na)
	  for n in 1:na
         a = arrays[n]
         # @belapsed extracts the runtime in seconds
         times[n] = @belapsed $func($a);
       end
       bm[!,name] = times
   end
   return bm
end

# ╔═╡ ecb61f9d-3af5-41dd-b38e-793e0b7b458f
sizes = [10^n for n in 2:5]

# ╔═╡ 18e13734-8ea4-43bf-b9fd-7177bbb29f2d
bm = benchmark(sizes)

# ╔═╡ a297af59-d3b5-4a7b-a3b9-a9b0b112ec84
plot(sizes, [bm[!, "slowsort"] bm[!, "quantile"]], linewidth=3,
    labels=["slowsort" "quantile"], legend=:top,
    markershape=:diamond, xscale=:log10, yscale=:log10, 
    xlabel="data size", ylabel="computing time (seconds)",
    title="Comparison of Sorting Algorithms")

# ╔═╡ 42d4ac31-4b5f-45e3-bae6-cecb4a0619dd
md"""
Add your favorite implementation of the quicksort algorithm to the plot.
"""

# ╔═╡ b5bbd617-d31d-4acf-a905-35af2496156f
md"""
## Connection with optimization

We can frame the averaging attribute as a solution to the following problem: find a number $a$ such that $a \approx a_n$ for $n=1,2,\ldots,N$. To solve this problem, we need a way to measure the approximation error. 
"""

# ╔═╡ c0cd50fb-46a5-408e-872c-3198c2335906
md"""
For example, we could take the *least-squares* or $L_2$ measure and minimize

$$F_{LS}(a) = \frac{1}{2}\,\sum\limits_{n=1}^N \left(a_n-a\right)^2\;.$$
"""

# ╔═╡ f913fe85-b439-4b9f-b6b9-e713c01a1605
md"""
$F_{LS}(a)$ is a quadratic function of $a$:

$$\begin{array}{rcl} F_{LS}(a) & = & \displaystyle \frac{1}{2}\,\sum\limits_{n=1}^N \left(a_n^2-2\,a_n\,a+a^2\right) \\ & =  &
\displaystyle \frac{1}{2}\,\sum\limits_{n=1}^N a_n^2 - a\,\sum\limits_{n=1}^N a_n + \frac{a^2}{2}\,N\;.\end{array}$$
"""

# ╔═╡ baa0a454-05be-4d95-b9a1-6b1dce265cec
md"""
It reaches the minimum value when

$${\frac{d}{d a} F_{LS}(a)} = {-\sum\limits_{n=1}^N a_n + a\,N} = 0\;.$$

or

$$a = {{\frac{1}{N}}}\,{{\sum\limits_{n=1}^{N} a_n}} = {\alpha}\;.$$
"""

# ╔═╡ 8234254e-6ee7-48b4-be0e-cb5a72284a51
md"""
Thus, the mean value corresponds to minimizing the misfit between an average value and data values measured with the least-squares misfit function $F_{LS}(a)$.
"""

# ╔═╡ f4e21ba5-7be6-41d3-8609-6806d642ba27
md"""
What if, instead of the $L_2$ measure, we use the sum of absolute values or the $L_1$ measure? The $L_1$ norm is defined as follows:

$$F_{L_1}(a) = \sum\limits_{n=1}^N \left|a_n-a\right|\;.$$
"""

# ╔═╡ b30a67c0-163e-45f4-9acf-6ab0762f40ef
md"""
The derivative of $F_{L_1}(a)$ involves the sum of sign functions

$${\frac{d}{d a} F_{L_1}(a)} = \displaystyle \sum\limits_{n=1}^N \frac{a-a_n}{ \left|a_n-a\right|} = \sum\limits_{n=1}^N \mbox{sign}(a-a_n)\;.$$

To set the sum of signs to zero, we need to ensure an equal number of positive and negative signs, or, in other words, the same number of data values greater than $a$ as less than $a$. 
"""

# ╔═╡ 30d2173c-d84a-4584-b648-1604f44df94b
md"""
According to our previous definition, this indicates that the optimal $a$ in the $L_1$ minimization case is the median.
"""

# ╔═╡ 6f9739db-d530-498a-81d0-a59c2963cd71
begin
	L1(data, a) = sum(abs.(data .- a))
	L1test(a) = L1(test, a)
end

# ╔═╡ 223d5ae5-cf2d-42aa-80ac-fe07d51ff658
begin
	L2(data, a) = sum((data .- a).^2)/2
	L2test(a) = L2(test, a)
end

# ╔═╡ 014c75d5-88b3-4acf-9d94-ca449c99f535
begin
	p1 = plot(L1test, 1, 100, 
	    label=L"$L_1$ objective function");
	p2 = plot(L2test, 1, 100, 
	    label=L"$L_2$ objective function", color=:green);
	plot(p1, p2, layout=(1,2), linewidth=3, xlabel=L"a", 
	    legend=:top)
end

# ╔═╡ 8228a8d9-5a51-4701-a3be-3d1bbb864c81
begin
	pz1 = plot(L1test, 46, 55, 
	          label=L"$L_1$ objective function");
	pz2 = plot(L2test, 46, 55, 
	          label=L"$L_2$ objective function", color=:green);
	plot(pz1, pz2, layout=(1,2), linewidth=3, xlabel=L"a", 
     legend=:top)
end

# ╔═╡ 63ce4067-556f-469b-84cb-310fc3fc0657
md"""
We could generalize the optimization approach by considering the $L_p$ measure

$$F_{L_p}(a) = \frac{1}{p} \sum\limits_{n=1}^N \left|a_n-a\right|^p$$

with different values of $p$. 
"""

# ╔═╡ fd8875f3-9cea-44fd-aadc-2d674dec6cda
begin
	Lp(data, a, p) = sum(abs.(data .- a).^p)/p
	L05test(a) = Lp(test, a, 0.5)
	L15test(a) = Lp(test, a, 1.5)
end

# ╔═╡ a1f3f878-5e5a-4aa3-8375-c9a2b44f5e10
begin
	p3 = plot(L05test, 46, 55, color=:red,
	          label=L"$L_{0.5}$ objective function");
	p4 = plot(L15test, 46, 55, color=:orange,
	          label=L"$L_{1.5}$ objective function");
	plot(p3, p4, layout=(1,2), linewidth=3, xlabel=L"a", 
	     legend=:top)
end

# ╔═╡ 2efed429-d5f8-4197-be4b-2589917eb6b8
md"""
An appropriate value of $p$ should be greater than one because, with smaller values, the objective function shows non-convex behavior.
"""

# ╔═╡ dca37065-b111-4455-9602-3c6c7ff1a0bb
md"""
Why select a specific measure and, consequently, a specific definition of the averaging attribute? The statistical argument links this choice to the probability distribution of data values. 
"""

# ╔═╡ e40aa494-5b50-415d-8152-128181a1c517
md"""
If the data are distributed according to the normal (Gaussian) distribution,

$$f(a_n) = \frac{1}{\sigma\,\sqrt{2\pi}}\exp\left[-\frac{(a_n-a)^2}{2\,\sigma^2}\right]\;,$$

and are statistically independent, then their joint probability distribution is the product of exponents
"""

# ╔═╡ 79a3f06a-5f61-4364-bdf2-a17f183d6b9e
md"""
$$\begin{array}{l}f(a_1,a_2,\ldots\,a_n) = \\ \displaystyle \frac{1}{\left(2\pi\,\sigma^2\right)^{N/2}}\,
 \exp\left[-\frac{1}{2\,\sigma^2}\,\sum\limits_{n=1}^N (a_n-a)^2\right]\;.\end{array}$$

Looking for $a$ that maximizes the probability becomes equivalent to minimizing the least-squares function $F_{LS}(a)$.
"""

# ╔═╡ 3f286665-de57-49c5-957e-4dbae59fe6a3
md"""
A key feature of the Gaussian distribution is its rapid decay from the center. Very large or very small values can still occur, but with quickly decreasing probability. When the decay rate is slower, indicating a higher likelihood of outliers, the $L_2$ measure becomes less suitable than other measures. 
"""

# ╔═╡ a67c8e9a-9721-4a2e-a5ac-fa138aa25c19
md"""
For example, the exponential probability distribution

$$f(a_n) = \frac{1}{2\,\mu}\exp\left[-\frac{|a_n-a|}{\mu}\right]$$

would lead to the $L_1$ minimization $F_{L_1}(a)$ and, correspondingly, to the median.
"""

# ╔═╡ 1759329d-2d21-4b46-aa76-2082a4b7e2f2
md"""
!!! note 
    Maximizing the likelihood of a specific parameter creates an optimization problem for estimating that parameter. The choice of an objective function for optimization depends on the statistical assumptions about the data distribution.
"""

# ╔═╡ 9cd42665-1434-4c40-95fb-d497f517473a
md"""
## Averaging complex numbers

Different ways exist to compute an average of two or more complex numbers. An arithmetic average between $c_1$ and $c_2$ is $a=(c_1+c_2)/2$. A geometric average $g=(c_1\,c_2)^{1/2}$ may better preserve amplitudes of the signal.

The loss of amplitude in stacking complex signals with varying phases can serve as a valuable measure of phase alignment.
"""

# ╔═╡ 38895986-8d7a-4e21-a98c-5866b31f9166
md"""
## References

* Bentley, J., 1999, Programming pearls, 2nd ed.: Addison-Wesley Professional.
* Cormen, T. H., 2013, Algorithms unlocked: The MIT Press.
* Cormen, T. H., C. E. Leiserson, R. L. Rivest, and C. Stein, 2009, Introduction to algorithms, 3rd ed.: The MIT Press.
"""

# ╔═╡ 1411f4b8-ce11-4d4e-8dc6-4c63d2056b10
md"""
* Devore, J. L., 2015, Probability and statistics for engineering and the sciences, 9th ed.: Cengage Learning.
* Hoare, C. A. R., 1962, Quicksort: Computer Journal, 5, 10–15.    
* Knuth, D. E., 1998, Art of computer programming, Volume 3: Sorting and searching, 2nd ed.: Addison-Wesley Professional.
"""

# ╔═╡ a4c98e64-d067-4ea8-853a-96221db725a7
md"""
* Tarantola, A., 2004, Inverse problem theory and methods for model parameter estimation: SIAM.
"""

# ╔═╡ 4e6920c1-1342-4fc9-86de-406da8ac46ba
md"""
## Data example

In the data example, we will revisit the daily temperature data from Camp Mabry. Let’s load the data from the previously saved CSV file.
"""

# ╔═╡ 0201f600-e98a-40bc-bc86-2922a40d1554
mabry = DataFrame(CSV.File("mabry.csv"))

# ╔═╡ e406bb06-6f48-4316-b5b6-dc53e4cef0a8
plot(mabry.Date, mabry.Temperature, 
     line_z=mabry.Temperature, color=:coolwarm, 
     title="Maximum Daily Temperature at Camp Mabry",
     ylabel="Degrees Celsius", leg=false)

# ╔═╡ bad25349-7f5c-4474-ae9a-42feba568ee7
md"""
There are many fluctuations in the data. To better observe the trend, it helps to apply local averaging to smooth the data. We will first try median averaging in a sliding window.
"""

# ╔═╡ d4dab228-959e-439a-a670-09b0a50bb668
function smoothed(data, w, average)
    "smoothing data in windows of size w using average"
    n = length(data)
    smooth = similar(data)
    for i in 1:n
        k = max(0,min(n-w,i-w÷2-1))
        smooth[i] = average(@view data[k+1:k+w])
    end
    return smooth
end

# ╔═╡ 9cc1ac4c-e28f-4761-877a-79e77c1fc00c
mabry.Smoothed = smoothed(mabry.Temperature, 30, median)

# ╔═╡ 6c604ec0-f42b-46eb-a9e6-6f859aa55106
begin
	plot(mabry.Date, mabry.Temperature, label="original", 
	     alpha=0.5);
	plot!(mabry.Date, mabry.Smoothed, linewidth=2, 
	      title="Maximum Daily Temperature at Camp Mabry",
	      ylabel="Degrees Celsius", label="smoothed")
end

# ╔═╡ 45bf3a32-3c5b-497b-815e-79f9b63b59be
md"""
Smoothing separates signal from noise. We can extract noise and analyze it separately by plotting its histogram. 
"""

# ╔═╡ 07a6a172-49be-416b-8c21-1cecf897b079
noise = mabry.Temperature - mabry.Smoothed;

# ╔═╡ 1bfd4961-fe4e-4d96-9874-bf14d6d4dc04
histogram(noise, label=:none, xlabel="Degrees Celsius", 
          title="Noise Distribution")

# ╔═╡ 34029c15-a1a7-4032-aa4d-148279213ed8
md"""
See the [StatsPlots](https://github.com/JuliaPlots/Plots.jl/tree/v2/StatsPlots) documentation for more ideas on plotting averages. 
"""

# ╔═╡ 731600be-a34c-4ac0-b58e-5a6d6aef6a19
md"""
!!! assignment
    ## Task 5

    Does the shape of the noise distribution justify the use of the median for the averaging function? 

    Repeat the analysis, replacing the running median with the running mean, and compare the results.
"""

# ╔═╡ b52ab147-e312-4b7b-b8fa-afe920c183e3
md"""
!!! assignment
    ## Bonus Task

    Extract an average of the maximum daily summer temperature for different years in the Camp Mabry dataset and plot it. For this assignment, you can define summer as the period between June 1 and August 31. 
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUIExtra = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
BenchmarkTools = "~1.5.0"
CSV = "~0.10.16"
DataFrames = "~1.8.2"
HTTP = "~1.11.0"
LaTeXStrings = "~1.4.0"
Plots = "~1.41.6"
PlutoUIExtra = "~0.1.8"
PyCall = "~1.96.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "73410b93c75d4a72c88d6449561e514b4a6dac2e"

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

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "b19db3927f0db4151cb86d073689f2428e524576"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.2"

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

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

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

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

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
# ╟─097057c3-8fa7-4c78-b44d-4536c6129cd4
# ╟─6166a05a-310b-4b1d-95b3-275cea7cec94
# ╟─a43bef1c-c149-4d4e-8ff3-15e07dad1cf8
# ╟─b466ed32-1534-423b-9cdf-cdd71f3da08a
# ╟─e79cd5b6-edae-4238-92a8-c4ddb5a2d44d
# ╟─13f1c675-3138-414b-827e-a0c2043cca73
# ╟─03e24dff-1b3f-4651-91aa-a531b14931f4
# ╠═9a711e15-0f99-4c83-8472-b5767508276a
# ╠═b2d1e843-1efd-49fc-b30c-62d86093c2a4
# ╟─f04f5adc-c0d7-4b1c-afc2-342815d9c529
# ╟─a66b0645-ca58-48e9-8442-1fc2fe17bca0
# ╟─7c0f5540-b9ed-4d73-af37-462d7a245acc
# ╟─f375d3af-81a8-4636-943a-a341062139b8
# ╟─1453c4e4-540a-40e6-9893-60051b89b841
# ╟─b80342c8-c673-4215-ad1a-b03cdd205e4a
# ╟─a7898225-7f51-410a-8b70-b03288792683
# ╟─63dc5fd8-51ae-44ab-b938-eef4462e1d80
# ╠═fb214786-5adf-4709-a6d6-48923e81b3bb
# ╠═68904d0b-4cc7-4951-b3b1-d146ecfc93c9
# ╠═21edb74e-6434-4b45-b0ac-8755e90f5992
# ╠═ae9c3b2e-af40-4ad4-96a9-93490420842e
# ╠═23774536-d1f0-4f0e-ae46-cc15a5a6fa3c
# ╠═5eb96318-b142-4479-a6c8-d06a44a75087
# ╠═8fff5fd1-33d3-46a6-afe5-ed9a6ade5c67
# ╠═7acffe75-e56a-44f7-821a-b3641f90022a
# ╟─cb379364-095f-4ab0-9609-425313be3fa0
# ╟─c2225da8-7903-459f-8722-280475dbdc1d
# ╟─e0a3bd12-0fac-484a-986f-feb514d763d3
# ╟─f95a8a8d-ee00-4730-9959-75f2569563fe
# ╠═8bf5b38f-2fe3-4c14-9c8b-677195a1a726
# ╠═5d4561d7-4dec-46c1-ab23-8e0084023e6c
# ╠═cd713688-43e0-4e08-a2a4-b80c0c66ea9b
# ╠═9c6c910e-65ec-4ec5-8a7c-35289c8e70d7
# ╟─16639520-f0c2-4518-93d7-934f69615232
# ╟─a67e9dca-573c-4d81-a663-4661afe7c4c1
# ╟─1555a6bc-eaf2-47a3-ae90-c59c656fdcda
# ╠═f1c6c404-3bbf-4c20-b40d-e821c01d914c
# ╠═19d39c97-c18b-432f-a7b4-0ecf2c90a247
# ╟─ed06a444-82bd-4d15-82f5-639570e2fce7
# ╟─e3819bf5-3976-4cf0-a8bc-0477950f35ed
# ╟─254daf4a-73d4-4d8d-974a-5fbf467f904f
# ╟─f728dd3b-732e-4ea4-8848-a5177172cf37
# ╟─aa2cd69d-816b-4fa4-8dfe-22a946cb7c5c
# ╟─e8d7d3b0-2b16-4f5c-a3ad-4a30261cc58d
# ╟─90ee6345-bd84-453f-91b5-1a0a8a878d93
# ╠═5bfd4ea4-a67d-4643-9c5e-ce1cf62049f8
# ╠═f90209c1-e32c-4c00-8ce0-8d33c4904f0d
# ╠═46b2cd69-a94f-4f5d-8ada-b9348c946dc5
# ╟─9b206b00-0f15-415b-a4a7-f9b8a6c8cad1
# ╠═e2242cbc-752c-4b34-8f6d-4f807ccef8b9
# ╠═bb83acd2-27a7-4ef1-b373-3b3d0f341bc8
# ╠═759b2a47-ff95-4e73-99fb-c26497b9dcf6
# ╟─d74bbfaf-2b70-4c44-b64a-51ec771a14da
# ╠═89578615-9355-4193-b11a-c2456a20a184
# ╠═c885494b-e081-4042-bc99-cb76c9bffea3
# ╠═43b357f7-d48e-4641-999c-8913358f7527
# ╠═9b311d95-ecf6-48d7-bf66-da06f6c3ee4d
# ╟─f00594ad-2eeb-4f76-9797-fe6b2da32895
# ╠═da3878f5-6d52-4228-a076-4e27118cac01
# ╠═15fd0528-6304-4d0c-8ef1-41a451699b24
# ╠═831bdedf-72f5-4317-a56b-f167e850e991
# ╠═bdacf5de-2ce5-4bea-94a1-41a75df0b16d
# ╠═92d12878-d2df-4e1b-868f-f8979b1bcd58
# ╠═0352b81a-7f0b-49db-9e7c-f2cf69fa2bab
# ╠═cc8fbae9-c9d2-41e5-87a0-529fca6d9292
# ╠═57363200-b2f5-46af-b33d-ca2bd06d0462
# ╠═63463017-11a6-4e17-a3fb-71b47fb6ff7d
# ╠═c042c4a5-f03e-425d-b195-5918d2126dfb
# ╠═80353213-14cc-423f-accd-97efe9a1d0ac
# ╠═43447d2b-c1aa-4dae-b98d-df3ce412496e
# ╠═584ee3fc-2e04-4ab4-899d-4ddcf7289876
# ╠═85303736-a9ef-43fc-b1c1-d7babb04aec0
# ╟─8ff320e7-2637-4c32-bfdc-1f014a7cbd94
# ╠═42d340a6-8566-4b1f-849e-ce7243da8d64
# ╟─6d72741b-a9a6-4dc7-9341-647e56ce0b16
# ╠═83f8b722-c7bb-452f-9290-3850daf62cb9
# ╠═21f33ba9-d33c-4122-b47d-9cd30795c57b
# ╟─e104818b-4d93-44b9-8437-81a3376b7f3c
# ╠═75a30393-5683-4c7f-96dd-1f9aec648e0c
# ╟─d94dcc77-3626-4625-80c7-869cd49d5554
# ╠═82bfbdc9-5a74-42b2-99c6-d41cbb78fdf8
# ╟─7bbe4f4d-cc17-41d0-a167-f96eb8a23015
# ╠═6491eaf2-de10-4336-9ed1-5969836abdf8
# ╠═646d7090-0594-424b-82d1-f6a43433976d
# ╟─a7e8f665-78bf-4338-8ef8-5f69d59bce60
# ╠═82ec41fa-9a8f-451c-8a48-de688c8833eb
# ╟─a5c5139c-d0b5-4086-8930-55d41c23f064
# ╠═d1934ea3-d942-48b3-8197-3abc11c9afca
# ╠═fafba457-1edf-4064-a895-83c80f85b168
# ╠═3a8c90c5-fe07-4b0a-b33b-46e5443f47f8
# ╠═79913fef-f496-4b06-97cc-d063ee5e6f67
# ╟─6d096f2c-5876-4465-a257-219b7c825842
# ╠═01b4fc8c-c48c-4e90-b65c-c578170c282d
# ╟─24a2397d-d209-408b-acfd-1355afda0c23
# ╠═ee3c1632-1939-4b27-820a-f9f35d2f7196
# ╠═c8101efa-eee2-4a20-b8c4-18c2ad6d44a2
# ╠═677b4943-c5ca-4253-9744-0957d4ded983
# ╠═159b140f-aee2-4d49-aa9c-60f679f448b4
# ╟─80a13b89-0729-40c9-8368-cb5afa35394b
# ╠═da2a9350-2a28-4fc3-84a6-76241a36f045
# ╠═fe8ef499-7bd8-443f-a303-f403ae4b9116
# ╠═9c057265-c7f8-40b1-bf43-87ad3fce5ff3
# ╠═ef73ca95-7c99-4fe2-9cba-4cb4cc0a8459
# ╟─e93be60d-5a9c-4dee-9155-c9be61c29fc2
# ╠═da1a4879-61cd-4263-b675-89b4c07aa91a
# ╠═ecb61f9d-3af5-41dd-b38e-793e0b7b458f
# ╠═18e13734-8ea4-43bf-b9fd-7177bbb29f2d
# ╠═a297af59-d3b5-4a7b-a3b9-a9b0b112ec84
# ╟─42d4ac31-4b5f-45e3-bae6-cecb4a0619dd
# ╟─b5bbd617-d31d-4acf-a905-35af2496156f
# ╟─c0cd50fb-46a5-408e-872c-3198c2335906
# ╟─f913fe85-b439-4b9f-b6b9-e713c01a1605
# ╟─baa0a454-05be-4d95-b9a1-6b1dce265cec
# ╟─8234254e-6ee7-48b4-be0e-cb5a72284a51
# ╟─f4e21ba5-7be6-41d3-8609-6806d642ba27
# ╟─b30a67c0-163e-45f4-9acf-6ab0762f40ef
# ╟─30d2173c-d84a-4584-b648-1604f44df94b
# ╠═6f9739db-d530-498a-81d0-a59c2963cd71
# ╠═223d5ae5-cf2d-42aa-80ac-fe07d51ff658
# ╠═45a88f15-932d-45ac-a12b-a99a880c1f26
# ╠═014c75d5-88b3-4acf-9d94-ca449c99f535
# ╠═8228a8d9-5a51-4701-a3be-3d1bbb864c81
# ╟─63ce4067-556f-469b-84cb-310fc3fc0657
# ╠═fd8875f3-9cea-44fd-aadc-2d674dec6cda
# ╠═a1f3f878-5e5a-4aa3-8375-c9a2b44f5e10
# ╟─2efed429-d5f8-4197-be4b-2589917eb6b8
# ╟─dca37065-b111-4455-9602-3c6c7ff1a0bb
# ╟─e40aa494-5b50-415d-8152-128181a1c517
# ╟─79a3f06a-5f61-4364-bdf2-a17f183d6b9e
# ╟─3f286665-de57-49c5-957e-4dbae59fe6a3
# ╟─a67c8e9a-9721-4a2e-a5ac-fa138aa25c19
# ╟─1759329d-2d21-4b46-aa76-2082a4b7e2f2
# ╟─9cd42665-1434-4c40-95fb-d497f517473a
# ╟─38895986-8d7a-4e21-a98c-5866b31f9166
# ╟─1411f4b8-ce11-4d4e-8dc6-4c63d2056b10
# ╟─a4c98e64-d067-4ea8-853a-96221db725a7
# ╟─4e6920c1-1342-4fc9-86de-406da8ac46ba
# ╠═0be18d16-10f0-4e8b-9876-3ee5d95cb4d3
# ╠═0201f600-e98a-40bc-bc86-2922a40d1554
# ╠═e406bb06-6f48-4316-b5b6-dc53e4cef0a8
# ╟─bad25349-7f5c-4474-ae9a-42feba568ee7
# ╠═d4dab228-959e-439a-a670-09b0a50bb668
# ╠═9cc1ac4c-e28f-4761-877a-79e77c1fc00c
# ╠═6c604ec0-f42b-46eb-a9e6-6f859aa55106
# ╟─45bf3a32-3c5b-497b-815e-79f9b63b59be
# ╠═07a6a172-49be-416b-8c21-1cecf897b079
# ╠═1bfd4961-fe4e-4d96-9874-bf14d6d4dc04
# ╟─34029c15-a1a7-4032-aa4d-148279213ed8
# ╟─731600be-a34c-4ac0-b58e-5a6d6aef6a19
# ╟─b52ab147-e312-4b7b-b8fa-afe920c183e3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
