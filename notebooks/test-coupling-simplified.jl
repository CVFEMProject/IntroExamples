### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 784b4c3e-bb2a-4940-a83a-ed5e5898dfd4
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using Revise
    using LinearAlgebra
    using CVFEMSystems: CVFEMSystem, solve, spacedim
    using CVFEMSystems:
        coord, transmission, nnodes, nedges, nodevolume, edgenode, dirichlet!
    using ExtendableGrids: simplexgrid, bfacemask!

    using GridVisualize:
        scalarplot, default_plotter!, scalarplot!, GridVisualizer, reveal, gridplot
    import CairoMakie
    default_plotter!(CairoMakie)
    using PlutoUI
end

# ╔═╡ a6c1ef20-16aa-40f0-82f6-2376126eefb7
md"""
In this notebook, we avoid the definition of an extra data structure, instead, we pass the Λ matrix directly as userdata.
"""

# ╔═╡ 9d5c39da-4461-4296-80ff-31bd3206d847
md"""
# The XT system
"""

# ╔═╡ f62010ad-6531-470e-8cad-c4ee682a0eac
md"""
```math
\begin{align}
∂_t X - ∇ ⋅  \left(a(x) ∇ X\right) &=0\\
∂_t T - ∇ ⋅ \left(φ(T) ∇ T + T ∇ X\right) &=0
\end{align}
```
"""

# ╔═╡ ce013ee5-4b2b-4982-b285-7c28c7b160b6
begin
    const iX = 1
    const iT = 2
    const Γ_in = 1
    const Γ_out = 2
end;

# ╔═╡ 431df177-6984-4132-91c8-11915246867d
φ(T) = 0.01 + 0.01 * T^2

# ╔═╡ 307164ca-3dbe-48a5-9bab-2bb14adea8a5
a(x) = 1

# ╔═╡ c72ee588-f1f2-46a0-a49f-845d108752c8
function celleval(y, u, celldata, userdata)
    Λ = userdata
    (; uold, tstep) = celldata
    ω = nodevolume(celldata)
    aavg = 0.0
    φavg = zero(eltype(u))
    for il = 1:nnodes(celldata)
        y[iX, il] += ((u[iX, il] - uold[iX, il]) / tstep) * ω
        y[iT, il] += ((u[iT, il] - uold[iT, il]) / tstep) * ω
        aavg += a(coord(celldata, il))
        φavg += φ(u[iT, il])
    end
    aavg /= nnodes(celldata)
    φavg /= nnodes(celldata)
    ΛKL = transmission(celldata, Λ)
    for ie = 1:nedges(celldata)
        i1 = edgenode(celldata, 1, ie)
        i2 = edgenode(celldata, 2, ie)

        dX = ΛKL[ie] * (u[iX, i1] - u[iX, i2])
        gX = aavg * dX
        y[iX, i1] += gX
        y[iX, i2] -= gX

        Tup = dX > 0 ? u[iT, i1] : u[iT, i2]
        gT = ΛKL[ie] * φavg * (u[iT, i1] - u[iT, i2]) + Tup * dX
        y[iT, i1] += gT
        y[iT, i2] -= gT
    end
end

# ╔═╡ 71a22381-b926-474d-91a5-fbcca08a3e1f
function bnodeeval(y, u, bnodedata, userdata)
    (; region) = bnodedata
    if region == Γ_in
        dirichlet!(bnodedata, y, u, 1.0; ispec = iX)
        dirichlet!(bnodedata, y, u, 1.0; ispec = iT)
    end
    if region == Γ_out
        dirichlet!(bnodedata, y, u, 0.0; ispec = iX)
        dirichlet!(bnodedata, y, u, 0.0; ispec = iT)
    end
end

# ╔═╡ 69f8af2f-af08-41c0-beae-b60b9737aa92
grid1d = simplexgrid(range(0, 1, length = 201))

# ╔═╡ cfb631fb-7aa8-4e09-9d0b-63a5ffa58bd7
sys1d = CVFEMSystem(grid1d, celleval, bnodeeval, ones(1, 1) , 2)

# ╔═╡ 7ff179f2-301c-4ebb-b1bc-d5d9a34e1ff9
solve(sys1d)

# ╔═╡ 90447221-fddb-4749-9453-0751f96825d0
Δt = 0.01

# ╔═╡ 04764f80-cf1e-488c-b278-eddd7b4ca32e
tsol1d = solve(sys1d; times = 0:Δt:1)

# ╔═╡ 08c64a57-3514-49eb-95d9-18296881273a
@bind it1d PlutoUI.Slider(1:length(tsol1d.t), show_value = true, default = 10)

# ╔═╡ 1058d158-5684-484b-b4c0-6fa857500395
let
    vis = GridVisualizer(
        size = (700, 300),
        legend = :rt,
        flimits = (-0.1, 2),
        title = "t=$(tsol1d.t[it1d])",
   )
    sol = tsol1d[it1d]
    scalarplot!(
        vis,
        grid1d,
        sol[iX, :],
        color = :green,
        label = "X"
    )
    scalarplot!(vis, grid1d, sol[iT, :], color = :red, label = "T", clear = false)
    reveal(vis)
end

# ╔═╡ 9fc8702c-1fe7-404b-a61d-39d23eaf6c78
begin
    grid2d = simplexgrid(range(0, 1, length = 201), (0:0.05:0.2))
    bfacemask!(grid2d, (0, 0), (1, 0), 3)
    bfacemask!(grid2d, (0, 0.2), (1, 0.2), 3)
    bfacemask!(grid2d, (0, 0), (0, 1), Γ_in)
    bfacemask!(grid2d, (1, 0), (1, 1), Γ_out)
end

# ╔═╡ f1cfa9f7-542c-43bc-9a35-3e67e23e4833
gridplot(grid2d, size = (700, 300), linewidth = 0.1)

# ╔═╡ f30503ea-6262-423c-870a-e6c2afed56e4
sys2d = CVFEMSystem(grid2d, celleval, bnodeeval, Matrix(Diagonal(ones(2))), 2)

# ╔═╡ 43d74112-72eb-4f28-91ca-6fb981b19d8b
sol2d = solve(sys2d)

# ╔═╡ 87b03221-ce70-43d3-803e-f12a6fd7b281
tsol2d = solve(sys2d; times = 0:0.01:1)

# ╔═╡ 00c3db40-11f4-4f4a-ba98-ecc949d02448
@bind it2d PlutoUI.Slider(1:length(tsol2d.t), show_value = true)

# ╔═╡ e6afb952-be58-4113-bba5-fba39747d7a9
let
    vis = GridVisualizer(
        size = (700, 200),
        layout = (1, 2),
        legend = :rt,
        flimits = (-0.1, 2),
     )
    sol = tsol2d[it2d]
    scalarplot!(vis[1, 1], grid2d, sol[iX, :], title = "X")
    scalarplot!(vis[1, 2], grid2d, sol[iT, :], title = "T")
    reveal(vis)
end

# ╔═╡ 1c63e831-ff08-46db-874e-6b2214b7a1c5
TableOfContents()

# ╔═╡ Cell order:
# ╠═784b4c3e-bb2a-4940-a83a-ed5e5898dfd4
# ╟─a6c1ef20-16aa-40f0-82f6-2376126eefb7
# ╟─9d5c39da-4461-4296-80ff-31bd3206d847
# ╟─f62010ad-6531-470e-8cad-c4ee682a0eac
# ╠═ce013ee5-4b2b-4982-b285-7c28c7b160b6
# ╠═431df177-6984-4132-91c8-11915246867d
# ╠═307164ca-3dbe-48a5-9bab-2bb14adea8a5
# ╠═c72ee588-f1f2-46a0-a49f-845d108752c8
# ╠═71a22381-b926-474d-91a5-fbcca08a3e1f
# ╠═69f8af2f-af08-41c0-beae-b60b9737aa92
# ╠═cfb631fb-7aa8-4e09-9d0b-63a5ffa58bd7
# ╠═7ff179f2-301c-4ebb-b1bc-d5d9a34e1ff9
# ╠═90447221-fddb-4749-9453-0751f96825d0
# ╠═04764f80-cf1e-488c-b278-eddd7b4ca32e
# ╠═08c64a57-3514-49eb-95d9-18296881273a
# ╠═1058d158-5684-484b-b4c0-6fa857500395
# ╠═9fc8702c-1fe7-404b-a61d-39d23eaf6c78
# ╠═f1cfa9f7-542c-43bc-9a35-3e67e23e4833
# ╠═f30503ea-6262-423c-870a-e6c2afed56e4
# ╠═43d74112-72eb-4f28-91ca-6fb981b19d8b
# ╠═87b03221-ce70-43d3-803e-f12a6fd7b281
# ╠═00c3db40-11f4-4f4a-ba98-ecc949d02448
# ╟─e6afb952-be58-4113-bba5-fba39747d7a9
# ╠═1c63e831-ff08-46db-874e-6b2214b7a1c5
