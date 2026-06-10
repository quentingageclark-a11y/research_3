### A Pluto.jl notebook ###
# v0.20.27

#> [frontmatter]
#> chapter = 4
#> section = 2
#> order = 2
#> image = "https://raw.githubusercontent.com/fonsp/Pluto.jl/580ab811f13d565cc81ebfa70ed36c84b125f55d/demo/plutodemo.gif"
#> title = "Second Data Analysis Notebook"
#> tags = ["module4", "track_julia", "track_material", "Pluto", "PlutoUI"]
#> layout = "layout.jlhtml"


using Markdown
using InteractiveUtils

# ╔═╡ ff7b5b69-7c8a-41ae-86f2-1fa73b1326f5
begin
	using PlutoUIExtra
	TableOfContents()
end

# ╔═╡ 80c83e60-ce53-4063-859e-3688024470fc
using Plots

# ╔═╡ e81c1ed6-d585-4e08-b45b-1c64b108e571
using CSV, HTTP, DataFrames

# ╔═╡ cd0cbaca-7eb0-4d48-a53b-9711f8d15d6e
using Dates

# ╔═╡ c92a07e0-bd7f-49d0-9e6f-514694fa0572
md"""
# Digital Data Representation

How are digital data stored in a computer? Before working with data examples, it helps to understand the basics of how continuous signals are represented digitally.
"""

# ╔═╡ 0b920e58-c299-4f07-8d35-106fc9f5c059
md"""
## Bits and bytes

An elementary unit of information is a *bit*: a state with two possible values ("yes" or "no", 1 or 0). Eight bits make up one *byte*. One byte of information can represent $2^{8}=256$ different values. This is enough to encode characters from the English alphabet (26 lowercase + 26 uppercase letters) + 10 digits + other symbols on a computer keyboard + several special symbols. 
"""

# ╔═╡ 0d9158ca-684d-45ea-ab28-f0e3aae20990
md"""
ASCII (American Standard Code for Information Interchange) is a standard system for mapping $2^{7}=128$ symbols, including 95 printable characters and 33 non-printing control characters. Regular text files are sometimes called ASCII files. 
"""

# ╔═╡ a4328d57-c066-4c81-8561-6fe771036b41
# print ASCII table

for i in 0:127
    println(i, " -> ", Char(i))
end

# ╔═╡ 1b2559b4-dcfa-4cc8-9ed8-3fb1b74a3c57
md"""
UTF-8 (Universal Coded Character Set Transformation Format – 8-bit) supported by Julia extends ASCII by using up to 4 bytes to represent over a million symbols. For example, the symbol with code 2025 is "ߩ", a character from the [N'Ko](https://everything2.com/title/NKo) writing system, which is used for the Manding languages of West Africa.
"""

# ╔═╡ b44f1d25-1190-4699-93b3-e3d629cb8ed1
println(Char(2025))

# ╔═╡ 9a1c8730-1c22-4b6b-81cd-a52c2f771afe
md"""
The Julia programming language provides an `Int8` data type for representing one-byte (8-bit) numbers, which range from -128 to 127. The corresponding *unsigned* data type `UInt8` also uses one byte but covers the range from 0 to 255. This is usually sufficient for representing 256 shades of gray, from "black" ($0$) to "white" ($255$). A more advanced color table with $2^{24}$ entries (over 16 million colors) uses three bytes to encode RGB (red, green, and blue) components.  
"""

# ╔═╡ 1b2e9e2f-c9dd-4ccd-81bb-7f1a182c29e9
function minmax(Type::DataType)
    println(" min($Type)=", typemin(Type),
            "\tmax($Type)=", typemax(Type))
end

# ╔═╡ a1c80b62-90cc-47ae-82af-1c2a4adff19c
minmax(Int8)

# ╔═╡ a7da6a9a-88eb-4069-9e23-96e419d80309
minmax(UInt8)

# ╔═╡ 57fe1cb2-7b4a-41f9-a113-4ef2c3f08112
md"""
!!! assignment
    ## Task 1

    UT Austin's official "burnt orange" color is represented by the code `#BF5700`, where each pair of symbols (`BF`, `57`, and `00`) corresponds to the hexadecimal (base 16) value of the RGB (red, green, and blue) components. Convert these numbers to their octal (base 8) and decimal (base 10) equivalents. 
"""

# ╔═╡ cb73ad25-19bc-472d-a942-6cd6882b5cca
plot(ann=[(0.5, 0.5, "Hook 'em Horns")], 
     showaxis=false, ticks=false,
     annotationfontsize=42, 
     annotationfontfamily="Georgia",    
     bg=RGB(0xBF/255, 0x57/255, 0x00/255))

# ╔═╡ e9af7905-d4bb-4ea1-b053-baed22d14ec6
md"""
!!! assignment
    ## Task 2

    The Julia function `characters()`, defined below, takes a string and prints its characters one by one. Modify this function to output the UTF-8 integer codes for each character in the string.
"""

# ╔═╡ 14828e18-22d7-4c8a-9f76-d014be5bab95
function characters(string::String)
    for char in string
        println(char)
    end
end

# ╔═╡ 9fb48768-afbb-496b-a144-b31797b3e458
characters("número")

# ╔═╡ 8c94eeb6-667a-41da-86f5-88e40db0ec51
md"""
## Integer data

We can represent integer values as text, but storing them in binary format is more efficient. For example, it takes at least 4 bytes (32 bits) to write the integer 2025 as the ASCII text "2025," but only 11 bits to represent it in binary.
"""

# ╔═╡ ecc96f54-2a1f-4e41-ab0f-4d3de1b14090
md"""
We need more than one byte of storage to hold integer values beyond the 0-255 range. In the Julia programming language, the type of an integer depends on the number of bytes allocated. There are five types of signed integers and five types of unsigned integers.
"""

# ╔═╡ 76f29851-7f3b-48fd-8ece-7de3cb2611b4
subtypes(Integer)

# ╔═╡ 98106f3f-dfc3-4fcd-bee8-1bdbd6e85fee
subtypes(Unsigned)

# ╔═╡ 10ca9298-13e2-4689-96e1-cc20c09a92fd
subtypes(Signed)

# ╔═╡ 99eeaaeb-5597-435f-b6f0-e576187d8222
md"""
Aside from `BigInt`, which uses adaptive precision, the other integer types specify the number of bits explicitly.
"""

# ╔═╡ 33a1377c-6407-4525-b8e0-c91d7b9c2c93
md"""
Consider a 4-byte (32-bit) integer. If it is unsigned, its value will range from $0$ to $2^{32}-1 \approx 4.3 \times 10^{9}$. If it is signed, its value will range from $-2^{31}$ to $2^{31}-1 \approx 2.15 \times 10^{9}$. A common way to encode signed integers is called *two's complement* and is shown in the table below.
"""

# ╔═╡ 3eb9eb89-2843-4f1d-8e6d-fc27f8e156c6
md"""
|      |   |   |   |      |       |      |        |       |       |   |   
|-----:|--:|--:|--:|-----:|------:|-----:|-------:|------:|------:|--:|
| **unsigned** | 0 | 1 | 2 | $\cdots$ | $2^{31}-1$ | $2^{31}$  | $2^{31}+1$  | $\cdots$ | $2^{32}-2$ | $2^{32}-1$ |
| **signed**   | 0 | 1 | 2 | $\cdots$ | $2^{31}-1$ | $-2^{31}$ | $-2^{31}+1$ | $\cdots$ | $-2$ | $-1$ |
"""

# ╔═╡ 920e1846-e39b-4ba4-b16c-43e5d12691de
intmax = typemax(Int32)

# ╔═╡ 237fe056-2ae4-45af-ac82-ab7e79aa91f6
for i::Int32 in -5:5
    println(intmax+i)
end

# ╔═╡ 69941e00-6fa5-4eba-8a46-2f2c389d437d
md"""
The main challenge of working with integer data is the existence of minimum and maximum values for each specific integer type. Another challenge is discussed in the next section.
"""

# ╔═╡ 968aa9e7-6a58-4f1f-9ab2-e427c26634e1
md"""
## Big endian and little endian

The terms *big-endian* and *little-endian* originate from Jonathan Swift's 18th-century satirical novel *Gulliver's Travels*. They refer to two different methods of cracking a boiled egg: from the big end or the little end. In *Gulliver's Travels*, this difference resulted in a war between two fictional nations.
"""

# ╔═╡ a1984924-28dc-4d4a-bd6c-f08c1902b73d
md"""
In computer applications, *big-endian* and *little-endian* refer to two ways of arranging a binary number: whether the most significant or least significant bytes come first. Like boiled eggs, the choice doesn't matter much, but it's important to stay consistent. 

* **Cohen, D., 1981. On holy wars and a plea for peace. Computer, 14(10), pp.48-54.**
"""

# ╔═╡ 20af1a72-0041-4960-8aa4-a477738770b2
md"""
Consider a 4-byte (32-bit) integer with a value of 2025. Since

$$\begin{array}{rcl}2025 & = & 1024 + 512 + 256 + 233 \\
& = & 1 \times 2^{10}+1 \times 2^9+1 \times 2^8+233\;,\end{array}$$

the binary representation of 2025 in the big-endian format will be

$$[0\cdots0][0\cdots0][0\cdots0111][\mbox{binary form of
  233}]\;.$$
"""

# ╔═╡ efe36a47-36c0-4065-ac73-297d9b596bc6
md"""
In the little-endian format, the same number will be represented as

$$[\mbox{binary form of
  233}][0\cdots0111][0\cdots0][0\cdots0]\;.$$
  
Incidentally,

$$\begin{array}{rcl} 233 & = & 128 + 64 + 32 + 8 + 1 \\ 
& = & 1 \times 2^7 + 1 \times 2^6 + 1 \times 2^5 +
1 \times 2^3 + 1 \times 2^0\;.\end{array}$$

Therefore, the binary form of $233$ is $11101001$. 
"""

# ╔═╡ fb35c483-c8b6-4ae6-864b-9bf5b8a85f3a
md"""
The choice of endianness depends on the hardware. Intel CPUs are
typically little-endian. 
"""

# ╔═╡ 8f0f2497-8dfc-44d4-b919-c61cc5db2950
md"""
The following Julia code converts a four-byte integer number into its
byte representation and prints it in the hex (base-16) format.
On a little-endian machine, the output is
```julia
2025 = UInt8[0xe9, 0x07, 0x00, 0x00]
```
where `e9` corresponds to the byte representing $233=14 \times
16 + 9$, and `07` corresponds to the $[0\cdots0111]$ byte.
"""

# ╔═╡ 7268e00c-c715-43d3-ac8f-dcfccd56501b
function bytes(number, Type::DataType)
    "find the binary representation of a given number"
    x = Type(number)
    println(number," = ",reinterpret(UInt8,[x]))
end

# ╔═╡ 72896cc3-bf67-4a20-b048-a9c526438b50
bytes(2025, Int32)

# ╔═╡ c1532040-fe3c-401d-964a-c3190a9a8028
md"""
## Floating-point data

The floating point representation for real numbers follows the
so-called *scientific notation*. In base 10, the typical
scientific notation is

$$\pm a \times 10^b\;,$$

where $a$ (mantissa) is such that $1 \le a < 10$, and $b$ (exponent) can be
positive or negative. For example, $2025$ in the scientific
notation is $+ 2.025 \times 10^3$. 
"""

# ╔═╡ 0ff3516f-8aa6-4e4e-852d-60a1e7de5c15
md"""
The scientific notation is also
called *floating point* because the point (dot) in the mantissa
is not fixed during arithmetic operations but rather "floats" to the front. 

In computer memory, it is more convenient to use a base-2 representation:

$$\pm a \times 2^b\;,$$

where $1 \le a < 2$.
"""

# ╔═╡ 7e612723-12b9-457d-b4a6-49ca9e165b7e
md"""
How are numbers like that stored in a computer? Several different encoding schemes exist. The most common ones are the IEEE single-precision (32-bit) and double-precision (64-bit) formats, established in 1985 by IEEE (the Institute of Electrical and Electronics Engineers). The table below shows how the bits are allocated in each case. 
"""

# ╔═╡ 8cab10a9-cb87-4916-aa9c-cbe3d7f30b74
md"""

|                           | sign | exponent | mantissa | total bits |
|:--------------------------|-----:|---------:|---------:|-----------:|
| **IEEE half-precision**   | 1 |  5 | 10 | 16 |
| **IEEE single-precision** | 1 |  8 | 23 | 32 |
| **IEEE double-precision** | 1 | 11 | 52 | 64 |

* **Overton, M.L., 2001. Numerical computing with IEEE floating point arithmetic. Society for Industrial and Applied Mathematics.**
"""

# ╔═╡ 042f4527-4c83-4cdd-8c14-58fd76b9ff4e
subtypes(AbstractFloat)

# ╔═╡ 650c9147-7c4a-4850-9568-cf4d12ca4a1a
md"""
The IEEE standard uses the first bit for the sign ($0$ for positive and $1$ for negative) and the following bits ($8$ for single-precision and $11$ for double-precision) for the exponent. The exponent is stored as an unsigned integer but is interpreted as a signed number after subtracting the bias value ($127$ for single precision and $1023$ for double precision). Therefore, in the single-precision case, the stored 8-bit number ranges from $0$ to $255$ but represents values from $-127$ to $128$ after subtracting the bias of $127$.  
"""

# ╔═╡ c2c54e55-faec-4a0a-bb83-ca32029bd9f8
md"""
One additional complication: the actual range of the exponent used for representing floating-point numbers is from $-126$ to $127$. The standard reserves extra values (stored as $0$ and $255$) to represent special numbers. 

Finally, the remaining bits ($23$ in the single-precision case and $52$ in the double-precision case) represent numbers after the decimal point in the mantissa. There is no need to store the number before the decimal point because it is always $1$ (the so-called *hidden bit*). 
"""

# ╔═╡ b23a2c2b-45ae-4177-b63f-ad1256fb1f92
md"""
What are the special numbers mentioned above? 

The first important number is $0$. The zero number does not have a trailing $1$. It uses $-127$ (stored as all $0$s) in the exponent and all $0$s in the mantissa. Interestingly, $+0.0$ and $-0.0$ are different floating-point numbers because of different sign bits (but they compare as equal in numerical computations).  
"""

# ╔═╡ fec90479-0542-4a61-b50f-a69274665310
md"""
The case of the exponent of all $0$s but the mantissa different from $0$
allows for representing additional numbers of the form $\pm a \times
2^{-126}$ with $0 < a < 1$. These *unnormalized* numbers increase
the precision of the floating-point system near zero. Thus,
the smallest positive floating-point number representable in single
precision is 

$2^{-23} \times 2^{-126} = 2^{-149} \approx 1.4 \times
10^{-45}\;.$
"""

# ╔═╡ e173dc56-a33a-40e8-be94-b816311e4705
md"""
Another special number is *infinity*. It occurs, for example, as the result of division by zero or overflow (exceeding the maximum or minimum representable floating-point values). Infinity is indicated by having the exponent of all 1s and the mantissa of all 0s. The sign bit distinguishes between $+\infty$ and $-\infty$. Representing infinity in the number system is useful because certain operations with infinities can be well-defined.
"""

# ╔═╡ 33ee666b-de63-4289-8460-15ab3f835c53
# instead of Float32(0) can use one(Float32)
a = Float32(1)/0 

# ╔═╡ 39e8057c-d8cd-48dc-9a85-92dd585ea751
typeof(Inf32)

# ╔═╡ aae63d19-26dc-4034-a31a-de2ba4488544
bytes(Inf32, Float32)

# ╔═╡ a744f530-1c44-4a94-806c-e9126c18ef9d
md"""
The first byte of `Int32` is 011111111 and corresponds to the number 127 (`0x7f` in hex). The second byte is 100000000 and corresponds to the number 128 (`0x80` in hex). With little-endian packing, these bytes appear last.
"""

# ╔═╡ a37f6f81-df68-4f55-ab39-f996bd0b9275
1 + Inf32

# ╔═╡ 498611ed-70b1-487f-b405-66077c906ba5
Inf32 + Inf32

# ╔═╡ c66e7146-a364-4ec1-89a7-5418023c7bf5
1/Inf32

# ╔═╡ c6bc09e2-6d94-4f91-a5e4-1ac430572385
md"""
Finally, an exceptional number is NaN (Not a Number). It occurs, for example, in the arithmetic operation $0.0/0.0$. NaNs have an exponent of all $1$s and a mantissa that is not zero. Different values in the mantissa enable the distinction among various types of NaNs. 
"""

# ╔═╡ 4d0a9973-b4e3-410c-9959-96b66b9f1746
# instead of Float32(0) can use zero(Float32)
b = Float32(0)/0 

# ╔═╡ ca400d5b-a3b0-4806-97c1-fc7820467893
bytes(NaN32, Float32)

# ╔═╡ 074a378c-a59f-4d96-97c9-86f7e47ccbf7
md"""
What is the largest number (less than infinity) that can be represented in a floating-point system? 

Let us consider the IEEE 64-bit (double-precision) system. In this system, 1 bit is reserved for the sign, 11 bits for the exponent, and 52 bits for the mantissa.
"""

# ╔═╡ 194304f9-3bb7-4f25-9e7c-58820d4f14ee
md"""
A double-precision normalized non-zero number $x$ can be written as 

$$x = \pm (1.d_1d_2{\cdots}d_{52})_2 \times 2^{n-1023}\;,$$

with $1 \le n \le 2046$, and $0 \le d_k \le 1$ for  $k=1,2,\ldots,52$. 
"""

# ╔═╡ aff5d0a1-a01d-42ae-971e-cb4233e94def
md"""
The largest finite number has all ones in the mantissa bits ($d_k=1$) and the maximum allowed number $n$ in the exponent. Therefore, it is equal to

$$\begin{array}{rcl} x_{\max} & = & \left(1+1/2+ \ldots + 1/2^{52}\right) \times 2^{2046-1023} \\ & = & (2-2^{-52}) \times 2^{1023} \\ & = & \left(1-2^{-53}\right) \times 2^{1024}\;.\end{array}$$

"""

# ╔═╡ c7516716-8000-4f8c-ba54-264865404c68
typemax(Float64)

# ╔═╡ e23a46d8-dcc9-44fb-a7c9-b694605628ea
max_float = prevfloat(Inf)

# ╔═╡ 35390cc7-0396-4f15-b41e-bab6cfde2e09
x::Float64 = (1-2^(-53))*2^1024 

# ╔═╡ 372e5d15-8919-4c86-99ed-1c5ddeb4c168
# use BigInt to avoid the integer overflow

x1::Float64 = (1-2^(-53))*BigInt(2)^1024 

# ╔═╡ 0001f79a-39ca-4df0-98f7-0ffbf2ebc292
# assert returns silently if the condition is true 
@assert(x1 == max_float) 

# ╔═╡ ed33b5c4-4734-4b0b-9c44-12813fa28130
@assert(2*2 == 4)

# ╔═╡ c72fcd54-cf26-4246-a3dd-ba9a18b4c336
@assert(2*2 == 5)

# ╔═╡ 96925321-7aaa-4028-8038-14cfe8fab9e3
xs::Float32 = (1-2^(-24))*BigInt(2)^128

# ╔═╡ eb54f30d-0828-480a-8a5b-11c8a7d4ec6a
@assert(xs == prevfloat(Inf32)) 

# ╔═╡ a17b2809-d29a-4b94-bb63-c39d78689bac
md"""
!!! assignment
    ## Task 3

    Find the largest number less than infinity in the IEEE 16-bit (half-precision) system and verify your derivation with Julia using `@assert`.
"""

# ╔═╡ 28d49463-7a0c-417b-8cf9-812f5ee357a6
md"""
Besides considering the smallest and largest numbers a floating-point system can represent (which addresses *underflow* and *overflow* issues), it is crucial to remember that the system is finite and contains gaps. No matter how precise the data representation, most real numbers cannot be exactly represented in a finite system.  
"""

# ╔═╡ 79baefd7-1d42-44a5-b97b-10638b08bf90
md"""
To describe the accuracy of the data representation, it is common to
use the *machine epsilon*: the smallest positive number
$\epsilon$ such that $1+\epsilon > 1$ in floating-point arithmetic. In
the IEEE single-precision format,

$$\epsilon=2^{-23} \approx 1.19 \times 10^{-7}\;.$$

In double precision,

$$\epsilon=2^{-52} \approx 2.22 \times 10^{-16}\;.$$
"""

# ╔═╡ 3d8c2038-032a-418a-bf12-6995cc074101
md"""
Therefore, it is commonly said that the single precision has between 6
and 7 significant (decimal) digits of accuracy, and the double precision has between 15 and 16 digits of accuracy. 

All numbers between $1$ and $1+\epsilon$ are missing from the floating-point system. Losing floating-point precision during numerical computations is a commonly occurring problem that requires attention and, occasionally, special treatment.
"""

# ╔═╡ 4ef5fef9-88ba-4b97-bad8-ada715bb2e1a
md"""
!!! note
    The main point about the floating-point system for representing real numbers in a computer is that it is finite. The system attempts to represent an infinite continuum of values but does so with limited precision.
"""

# ╔═╡ 70f7ac13-12b5-4d8a-9cd5-9a533718e4f1
md"""
!!! assignment
    ## Task 4

    Let us try to compute the *machine epsilon* directly. The following program does this but fails without an assertion error because of a missing line. Add the missing line to silence the `@assert`.
"""

# ╔═╡ fbb80708-d8a1-4a5a-92d8-1962426ec03b
function find_epsilon(Type::DataType)
	ε = one(Type) # number 1 

    while 1+ε > 1
        ε /= 2  # same as ε = ε/2
    end

    return 2 * ε  # multiply by 2 to get the machine epsilon
end

# ╔═╡ 054a2275-2728-4286-b805-d659da6caa31
println("machine epsilon (32-bit) is $(find_epsilon(Float32))")

# ╔═╡ e1ede4b8-203c-4571-95b0-104915fd0450
@assert(find_epsilon(Float32) == eps(Float32))

# ╔═╡ c98641da-2498-4f60-bfad-66c76141c436
@assert(find_epsilon(Float64) == eps(Float64))

# ╔═╡ 238e27ab-dd7e-4d5b-9b70-243680cc6a72
md"""
## Complex numbers

A complex number consists of two parts (real and imaginary). Therefore, it takes twice the storage of the corresponding real number.
"""

# ╔═╡ 31a5b4cd-db28-448b-9bfe-0aed06f6b99c
c = 1.0 + 0.0im  # More idiomatic complex literal

# ╔═╡ 4fb65e0e-fecb-44bf-a409-7e094c037a80
typeof(c)

# ╔═╡ 955327e3-6fb8-4a71-847b-9113c2b6e977
typeof(1+2im)

# ╔═╡ b805a4ab-c627-481b-9dcc-fedcfc200b3d
md"""
## Data example

The example dataset we will use includes measurements from the Camp Mabry weather station in Austin, TX. The data are publicly available and can be downloaded from the Global Historical Climatology Network (GHCN) maintained by the National Centers for Environmental Information (NCEI). 
"""

# ╔═╡ 6a597f80-077a-4849-a0e3-4ee8da547bd6
url="https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USW00013958.csv.gz"

# ╔═╡ 6b8a824b-6415-4971-8259-b4db73e5bf44
md"""
Instead of downloading the file (in compressed CSV format) and then trying to read it, we will load it directly into a data frame object using appropriate packages.
"""

# ╔═╡ 9ed90efa-abf8-4349-a505-ecee3030a50b
# read the data from url into a data frame
df = CSV.read(HTTP.get(url).body, DataFrame; header=0)

# ╔═╡ 23ea7729-d608-436f-a62a-8ca51d568e02
typeof(df)

# ╔═╡ b9887c02-231a-4685-b590-cd3e5a43a2a5
md"""
A data frame is a table-like object, which may contain values of different types and is convenient for data manipulation.

Table `df` contains different meteorological measurements from June 1, 1938 (identified by 19380601 in **Column2** in the first row) to the present. 
"""

# ╔═╡ 56c6d1fd-1935-410e-a912-55b6ba4354be
md"""
Let us reduce the dataset. We will extract from its measurements only one quantity (the maximum daily temperature identified as `TMAX` in **Column3**) and the dates after September 1, 2013. 
"""

# ╔═╡ a6b337db-33e8-415c-8ad8-641fb60e5114
data = df[(df.Column3 .== "TMAX") .& (df.Column2 .> 20130901), :]

# ╔═╡ fcead942-7184-4cb5-bb5f-ad6ce10d6ade
md"""
Let us keep only the columns we are interested in.
"""

# ╔═╡ d1cd0d0c-fa2b-48e1-8e9f-eb9634b659a9
data2 = select(data, [:Column2, :Column4]);

# ╔═╡ 06b82fb3-7b2a-467d-b022-4761fcc9fa51
# Display the first seven rows
first(data2, 7)

# ╔═╡ 20cbb9ab-d053-467c-a54a-34e89afaecaa
md"""
Let us give the two columns descriptive names.
"""

# ╔═╡ 9ac0aa11-4c27-4db8-bbba-a440a5e5ee3f
rename!(data2, :Column2 => :Date, :Column4 => :Temperature);

# ╔═╡ a00e983e-bf71-4eb4-ad2f-b284bd3f8001
first(data2, 7)

# ╔═╡ 25972d76-9f7e-48ef-b624-7f78333eb80c
describe(data2)

# ╔═╡ e148c409-1535-4c3c-a7db-a694d00f27e6
md"""
The data are stored as integers (type `Int64`), but this is incorrect. The date column contains dates, and the temperature column contains temperatures (Celsius with one decimal point). Let us convert them to a proper data type.
"""

# ╔═╡ b51c2f22-0c1e-49c0-84cc-613869b1e12a
date_format = DateFormat("yyyymmdd")

# ╔═╡ 64c3bbb8-eb77-4a38-9d9a-e86935be45f2
fixdate(date) = Date(string(date), date_format)

# ╔═╡ c759f490-46c4-49dc-9087-3e8c4f13e5bf
fixdate(20130902)

# ╔═╡ dba95c6b-cfd3-4530-a0bd-4c0757f79870
begin
	data2.Date = fixdate.(data2.Date)
	data2.Temperature = data2.Temperature ./ 10  # vectorized division
	data2
end

# ╔═╡ b85de9dd-1394-4686-8f79-b1f3a99adf30
describe(data2)

# ╔═╡ 88331b48-02d5-452b-98f5-965bca7dad0d
md"""
Now, we are ready to display our data.
"""

# ╔═╡ 5a8e491d-2521-4050-ae93-850526208d08
plot(data2.Date, data2.Temperature, line_z=data2.Temperature,
     title="Maximum Daily Temperature at Camp Mabry",
     ylabel="Degrees Celsius", color=:coolwarm, leg=false)

# ╔═╡ 19d39743-ef7c-459f-be7f-58d7c8c51e1f
md"""
!!! assignment
    ## Task 5

    Convert the temperature from Celsius to Fahrenheit and create a new plot.

    The formula for conversion is

    $\mbox{F} = \mbox{C} \times 9/5 + 32\;.$ 
"""

# ╔═╡ e8aff59d-ca7d-430a-916a-3fe8ec4271b0
md"""
!!! assignment
    ## Bonus Task

    For an extra challenge, design a different type of display that better shows annual temperature cycles. For example, you could use color to indicate the time of year, split the data by year to create a two-dimensional plot, use a polar plot with temperature as the radius, add animation, etc.  
"""

# ╔═╡ f6618677-cb88-4c76-9903-8b3a298e5a1d
# Save the dataset in a file

CSV.write("mabry.csv",data2)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUIExtra = "a011ac08-54e6-4ec3-ad1c-4165f16ac4ce"

[compat]
CSV = "~0.10.16"
DataFrames = "~1.8.2"
HTTP = "~1.11.0"
Plots = "~1.41.6"
PlutoUIExtra = "~0.1.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "84d1242fb27aeaa10e498c8c17122abc1dfdb194"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

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
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "fe23330af47b8ab4e135b2ff65f7398c3a2bfc65"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.5.2"

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
git-tree-sha1 = "fbc875044d82c113a9dee6fc14e16cf01fd48872"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.80"

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
git-tree-sha1 = "ebe7e59b37c400f694f52b58c93d26201387da70"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.9"

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

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "dd974aefe288ef2898733aecf40858dc86742d74"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.1"

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
git-tree-sha1 = "429722587208f02b1cecbddcd20133df2f1ed796"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+0"

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
# ╟─ff7b5b69-7c8a-41ae-86f2-1fa73b1326f5
# ╟─c92a07e0-bd7f-49d0-9e6f-514694fa0572
# ╟─0b920e58-c299-4f07-8d35-106fc9f5c059
# ╟─0d9158ca-684d-45ea-ab28-f0e3aae20990
# ╠═a4328d57-c066-4c81-8561-6fe771036b41
# ╟─1b2559b4-dcfa-4cc8-9ed8-3fb1b74a3c57
# ╠═b44f1d25-1190-4699-93b3-e3d629cb8ed1
# ╟─9a1c8730-1c22-4b6b-81cd-a52c2f771afe
# ╠═1b2e9e2f-c9dd-4ccd-81bb-7f1a182c29e9
# ╠═a1c80b62-90cc-47ae-82af-1c2a4adff19c
# ╠═a7da6a9a-88eb-4069-9e23-96e419d80309
# ╟─57fe1cb2-7b4a-41f9-a113-4ef2c3f08112
# ╠═80c83e60-ce53-4063-859e-3688024470fc
# ╠═cb73ad25-19bc-472d-a942-6cd6882b5cca
# ╟─e9af7905-d4bb-4ea1-b053-baed22d14ec6
# ╠═14828e18-22d7-4c8a-9f76-d014be5bab95
# ╠═9fb48768-afbb-496b-a144-b31797b3e458
# ╟─8c94eeb6-667a-41da-86f5-88e40db0ec51
# ╟─ecc96f54-2a1f-4e41-ab0f-4d3de1b14090
# ╠═76f29851-7f3b-48fd-8ece-7de3cb2611b4
# ╠═98106f3f-dfc3-4fcd-bee8-1bdbd6e85fee
# ╠═10ca9298-13e2-4689-96e1-cc20c09a92fd
# ╟─99eeaaeb-5597-435f-b6f0-e576187d8222
# ╟─33a1377c-6407-4525-b8e0-c91d7b9c2c93
# ╟─3eb9eb89-2843-4f1d-8e6d-fc27f8e156c6
# ╠═920e1846-e39b-4ba4-b16c-43e5d12691de
# ╠═237fe056-2ae4-45af-ac82-ab7e79aa91f6
# ╟─69941e00-6fa5-4eba-8a46-2f2c389d437d
# ╟─968aa9e7-6a58-4f1f-9ab2-e427c26634e1
# ╟─a1984924-28dc-4d4a-bd6c-f08c1902b73d
# ╟─20af1a72-0041-4960-8aa4-a477738770b2
# ╟─efe36a47-36c0-4065-ac73-297d9b596bc6
# ╟─fb35c483-c8b6-4ae6-864b-9bf5b8a85f3a
# ╟─8f0f2497-8dfc-44d4-b919-c61cc5db2950
# ╠═7268e00c-c715-43d3-ac8f-dcfccd56501b
# ╠═72896cc3-bf67-4a20-b048-a9c526438b50
# ╟─c1532040-fe3c-401d-964a-c3190a9a8028
# ╟─0ff3516f-8aa6-4e4e-852d-60a1e7de5c15
# ╟─7e612723-12b9-457d-b4a6-49ca9e165b7e
# ╟─8cab10a9-cb87-4916-aa9c-cbe3d7f30b74
# ╠═042f4527-4c83-4cdd-8c14-58fd76b9ff4e
# ╟─650c9147-7c4a-4850-9568-cf4d12ca4a1a
# ╟─c2c54e55-faec-4a0a-bb83-ca32029bd9f8
# ╟─b23a2c2b-45ae-4177-b63f-ad1256fb1f92
# ╟─fec90479-0542-4a61-b50f-a69274665310
# ╟─e173dc56-a33a-40e8-be94-b816311e4705
# ╠═33ee666b-de63-4289-8460-15ab3f835c53
# ╠═39e8057c-d8cd-48dc-9a85-92dd585ea751
# ╠═aae63d19-26dc-4034-a31a-de2ba4488544
# ╟─a744f530-1c44-4a94-806c-e9126c18ef9d
# ╠═a37f6f81-df68-4f55-ab39-f996bd0b9275
# ╠═498611ed-70b1-487f-b405-66077c906ba5
# ╠═c66e7146-a364-4ec1-89a7-5418023c7bf5
# ╟─c6bc09e2-6d94-4f91-a5e4-1ac430572385
# ╠═4d0a9973-b4e3-410c-9959-96b66b9f1746
# ╠═ca400d5b-a3b0-4806-97c1-fc7820467893
# ╟─074a378c-a59f-4d96-97c9-86f7e47ccbf7
# ╟─194304f9-3bb7-4f25-9e7c-58820d4f14ee
# ╟─aff5d0a1-a01d-42ae-971e-cb4233e94def
# ╠═c7516716-8000-4f8c-ba54-264865404c68
# ╠═e23a46d8-dcc9-44fb-a7c9-b694605628ea
# ╠═35390cc7-0396-4f15-b41e-bab6cfde2e09
# ╠═372e5d15-8919-4c86-99ed-1c5ddeb4c168
# ╠═0001f79a-39ca-4df0-98f7-0ffbf2ebc292
# ╠═ed33b5c4-4734-4b0b-9c44-12813fa28130
# ╠═c72fcd54-cf26-4246-a3dd-ba9a18b4c336
# ╠═96925321-7aaa-4028-8038-14cfe8fab9e3
# ╠═eb54f30d-0828-480a-8a5b-11c8a7d4ec6a
# ╟─a17b2809-d29a-4b94-bb63-c39d78689bac
# ╟─28d49463-7a0c-417b-8cf9-812f5ee357a6
# ╟─79baefd7-1d42-44a5-b97b-10638b08bf90
# ╟─3d8c2038-032a-418a-bf12-6995cc074101
# ╟─4ef5fef9-88ba-4b97-bad8-ada715bb2e1a
# ╟─70f7ac13-12b5-4d8a-9cd5-9a533718e4f1
# ╠═fbb80708-d8a1-4a5a-92d8-1962426ec03b
# ╠═054a2275-2728-4286-b805-d659da6caa31
# ╠═e1ede4b8-203c-4571-95b0-104915fd0450
# ╠═c98641da-2498-4f60-bfad-66c76141c436
# ╟─238e27ab-dd7e-4d5b-9b70-243680cc6a72
# ╠═31a5b4cd-db28-448b-9bfe-0aed06f6b99c
# ╠═4fb65e0e-fecb-44bf-a409-7e094c037a80
# ╠═955327e3-6fb8-4a71-847b-9113c2b6e977
# ╟─b805a4ab-c627-481b-9dcc-fedcfc200b3d
# ╠═6a597f80-077a-4849-a0e3-4ee8da547bd6
# ╟─6b8a824b-6415-4971-8259-b4db73e5bf44
# ╠═e81c1ed6-d585-4e08-b45b-1c64b108e571
# ╠═9ed90efa-abf8-4349-a505-ecee3030a50b
# ╠═23ea7729-d608-436f-a62a-8ca51d568e02
# ╟─b9887c02-231a-4685-b590-cd3e5a43a2a5
# ╟─56c6d1fd-1935-410e-a912-55b6ba4354be
# ╠═a6b337db-33e8-415c-8ad8-641fb60e5114
# ╟─fcead942-7184-4cb5-bb5f-ad6ce10d6ade
# ╠═d1cd0d0c-fa2b-48e1-8e9f-eb9634b659a9
# ╠═06b82fb3-7b2a-467d-b022-4761fcc9fa51
# ╟─20cbb9ab-d053-467c-a54a-34e89afaecaa
# ╠═9ac0aa11-4c27-4db8-bbba-a440a5e5ee3f
# ╠═a00e983e-bf71-4eb4-ad2f-b284bd3f8001
# ╠═25972d76-9f7e-48ef-b624-7f78333eb80c
# ╟─e148c409-1535-4c3c-a7db-a694d00f27e6
# ╠═cd0cbaca-7eb0-4d48-a53b-9711f8d15d6e
# ╠═b51c2f22-0c1e-49c0-84cc-613869b1e12a
# ╠═64c3bbb8-eb77-4a38-9d9a-e86935be45f2
# ╠═c759f490-46c4-49dc-9087-3e8c4f13e5bf
# ╠═dba95c6b-cfd3-4530-a0bd-4c0757f79870
# ╠═b85de9dd-1394-4686-8f79-b1f3a99adf30
# ╟─88331b48-02d5-452b-98f5-965bca7dad0d
# ╠═5a8e491d-2521-4050-ae93-850526208d08
# ╟─19d39743-ef7c-459f-be7f-58d7c8c51e1f
# ╟─e8aff59d-ca7d-430a-916a-3fe8ec4271b0
# ╠═f6618677-cb88-4c76-9903-8b3a298e5a1d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
