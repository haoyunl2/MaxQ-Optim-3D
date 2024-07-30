## Maximize injection rate within fracture pressure bound with gradient-based 
## backtracking line search optimization solver


## cd ~/ccs/MaxQ-Optim-3D
## julia --project=.


import Pkg; Pkg.add("DrWatson")
Pkg.instantiate()  # This installs all dependencies listed in Project.toml
Pkg.precompile()   # Precompile all installed packages

using DrWatson
@quickactivate "MaxQ-Optim-3D"

using LinearAlgebra
using SlimOptim
using Printf
using JLD2

using Jutul 
using JutulDarcy
using HYPRE
using GLMakie
using CSV
using DataFrames


nx = 31
ny = 30
nz = 10
Darcy, bar, kg, meter, day = si_units(:darcy, :bar, :kilogram, :meter, :day)

# Set up a 2D acquifer model
cart_dims = (nx, ny, nz)
physical_dims = (1000.0, 1000.0, 1000.0)
mesh = UnstructuredMesh(CartesianMesh(cart_dims, physical_dims))

# Read the xyz position from 
top = Matrix(CSV.read("../data/CCS2_small_xyz.dat", DataFrame; header=false))
base = Matrix(CSV.read("../data/CCS2_small_base_xyz.dat", DataFrame; header=false))

# using Geodesy

# Gulf of mexico is in UTM zone 15, north hemispehre
# utm_zone = UTMZ(15, true, wgs84)



# depth = base[1, 3] - base[(nx + 1) * (ny + 1), 3]
# # using Ranges
# depth_arr = LinRange(0.0, depth, 11) .- base[1, 3]


# depth_pair = zeros(2, (nx + 1) * (ny + 1))
# depth_pair[1, 1:(nx + 1) * (ny + 1)] = (-1) * top[1:(nx + 1) * (ny + 1), 3]
# depth_pair[2, 1:(nx + 1) * (ny + 1)] = (-1) * base[1:(nx + 1) * (ny + 1), 3]



points = mesh.node_points
for (i, pt) in enumerate(points)
    # x, y, z = pt
    # x_u = 2*π*x/1000.0
    # w = 0.2
    # dz = 0.05*x + w*(30*sin(2.0*x_u) + 20*sin(5.0*x_u) + 10*sin(10.0*x_u) + 5*sin(25.0*x_u))
    # points[i] = pt + [0, 0, dz]

    # xy_utm = base[i % ((nx + 1) * (ny + 1)) == 0 ? (nx + 1) * (ny + 1) : i % ((nx + 1) * (ny + 1)), :]
    # x_meters, y_meters = utm_to_meters(utm_zone, xy_utm[1], xy_utm[2])
    # points[i] = [x_meters, y_meters, depth_arr[i == (nx + 1) * (ny + 1) * (nz + 1)  ? nz + 1 : div(i, (nx + 1) * (ny + 1)) + 1]]
    
    xyz_top = top[i % ((nx + 1) * (ny + 1)) == 0 ? (nx + 1) * (ny + 1) : i % ((nx + 1) * (ny + 1)), :]
    xyz_base = base[i % ((nx + 1) * (ny + 1)) == 0 ? (nx + 1) * (ny + 1) : i % ((nx + 1) * (ny + 1)), :]
    rank = div(i, (nx + 1) * (ny + 1)) == 11 ? 11 : div(i, (nx + 1) * (ny + 1)) + 1 
    depth_arr = LinRange((-1) * xyz_top[3], (-1) * xyz_base[3], 11) 
    # points[i] = [xyz_base[1] - base[1, 1], xyz_base[2] -  base[(nx + 1) * (ny + 1), 2], depth_arr[i == (nx + 1) * (ny + 1) * (nz + 1)  ? nz + 1 : div(i, (nx + 1) * (ny + 1)) + 1]]
    points[i] = [xyz_base[1] - base[1, 1], xyz_base[2] -  base[(nx + 1) * (ny + 1), 2], depth_arr[rank]]

end



# # 0.orginal universal permeability

# # sample 0
# sp = 0

# # setup the simulation model
# domain = reservoir_domain(mesh, permeability = 1.0Darcy, porosity = 0.3, temperature = convert_to_si(30.0, :Celsius))

# # Injector xy spatial position
# Injector_xy = [302600 - base[1, 1], 3176650 - base[(nx + 1) * (ny + 1), 2]]

# # Injector xyz coordinates
# Injector_cor = (Int(round((Injector_xy[1] - points[1][1]) / (points[nx + 2][1] - points[1][1]))), 
# Int(round((points[1][2] - Injector_xy[2]) / (points[1][2] - points[2][2]))),  nz - div(nz, 4))


# # Injector = setup_well(domain, (65, 1, 1), name = :Injector)
# Injector = setup_well(domain, (Injector_cor[1], Injector_cor[2], Injector_cor[3]), name = :Injector)
# model, parameters = setup_reservoir_model(domain, :co2brine, wells = Injector);





## 1.create the permeability with several layers

# sample 1
sp = 1

perm1 = ones(nx, ny, nz) * 1.0Darcy
# perm1[:, :, 1] *= 0.0
perm1[:, :, 1] *= 1 / 1000
# perm1[:, :, nz] *= 0.02 / 1000
perm1[:, :, nz] *= 40 / 1000
perm1[:, :, 3] .*= 40/1000
perm1[:, :, 5] .*= 40/1000
perm1[:, :, 7] .*= 40/1000


poro1 = ones(nx, ny, nz) * 0.27
# poro1[:, :, 1] .*= 0.0
poro1[:, :, 1] .*= 0.003/ 0.27
# poro1[:, :, nz] .*= 0.02/0.27
poro1[:, :, nz] .*= 0.11/0.27
poro1[:, :, 3] .*= 0.11/0.27
poro1[:, :, 5] .*= 0.11/0.27
poro1[:, :, 7] .*= 0.11/0.27

domain = reservoir_domain(mesh, permeability = vec(perm1), porosity = vec(poro1), temperature = convert_to_si(30.0, :Celsius))

Injector_xy = [302600 - base[1, 1], 3176650 - base[(nx + 1) * (ny + 1), 2]]

Injector_cor = (Int(round((Injector_xy[1] - points[1][1]) / (points[nx + 2][1] - points[1][1]))), 
Int(round((points[1][2] - Injector_xy[2]) / (points[1][2] - points[2][2]))), nz - div(nz, 4))


# Injector = setup_well(domain, (65, 1, 1), name = :Injector)
Injector = setup_well(domain, (Injector_cor[1], Injector_cor[2], Injector_cor[3]), name = :Injector)
model, parameters = setup_reservoir_model(domain, :co2brine, wells = Injector);





# ## 2.create the permeability with several layers and fault

# # sample 2
# sp = 2

# perm2 = ones(nx, ny, nz) * 1.0Darcy
# perm2[:, :, 1] *= 0.0
# perm2[:, :, nz] *= 0.02 / 1000
# perm2[:, :, 3] .*= 40/1000
# perm2[:, :, 5] .*= 40/1000
# perm2[:, :, 7] .*= 40/1000


# Injector_xy = [302600 - base[1, 1], 3176650 - base[(nx + 1) * (ny + 1), 2]]

# Injector_cor = (round((Injector_xy[1] - points[1][1]) / (points[nx + 2][1] - points[1][1])), 
# round((points[1][2] - Injector_xy[2]) / (points[1][2] - points[2][2])))


# perm2[Int(Injector_cor[1]) - 1:Int(Injector_cor[1]) + 2, Int(Injector_cor[2]) - 1, 2:10] = 20/1000 * 1.0Darcy * ones(4, 9)
# perm2[Int(Injector_cor[1]) - 1, Int(Injector_cor[2]) - 1:Int(Injector_cor[2]) + 2, 2:10] = 20/1000 * 1.0Darcy * ones(4, 9)

# poro2 = ones(nx, ny, nz) * 0.27
# poro2[:, :, 1] .*= 0.0
# poro2[:, :, nz] .*= 0.02/0.27
# poro2[:, :, 3] .*= 0.11/0.27
# poro2[:, :, 5] .*= 0.11/0.27
# poro2[:, :, 7] .*= 0.11/0.27

# poro2[Int(Injector_cor[1]) - 1:Int(Injector_cor[1]) + 2,  Int(Injector_cor[2]) - 1 , 2:10] = 0.05 * ones(4, 9)
# poro2[Int(Injector_cor[1]) - 1,  Int(Injector_cor[2]) - 1:Int(Injector_cor[2]) + 2, 2:10] = 0.05 * ones(4, 9)

# domain = reservoir_domain(mesh, permeability = vec(perm2), porosity = vec(poro2), temperature = convert_to_si(30.0, :Celsius))
# Injector = setup_well(domain, (Int(Injector_cor[1]), Int(Injector_cor[2]), nz - div(nz, 4)), name = :Injector)
# model, parameters = setup_reservoir_model(domain, :co2brine, wells = Injector);



# ## 3.create the permeability with several layers and baffles
# perm3 = ones(100, 50) * 1.0Darcy
# perm3[:, 1:2] *= 0.0
# perm3[:, 50] *= 0.02 / 1000
# perm3[:, 12] .*= 40/1000
# perm3[:, 22] .*= 40/1000
# perm3[:, 32] .*= 40/1000
# perm3[:, 42] .*= 40/1000

# perm3[40:45, 3:8] = 40 / 1000 * 1.0Darcy * ones(6, 6)
# perm3[55:60, 3:8] = 40 / 1000 * 1.0Darcy * ones(6, 6)
# perm3[70:75, 3:8] = 40 / 1000 * 1.0Darcy * ones(6, 6)
# perm3[80:85, 3:8] = 40 / 1000 * 1.0Darcy * ones(6, 6)

# perm3[50:53, 13:16] = 40 / 1000 * 1.0Darcy * ones(4, 4)
# perm3[60:63, 13:16] = 40 / 1000 * 1.0Darcy * ones(4, 4)
# perm3[68:71, 13:16] = 40 / 1000 * 1.0Darcy * ones(4, 4)


# perm3[20:23, 23:30] = 40 / 1000 * 1.0Darcy * ones(4, 8)
# perm3[70:73, 23:30] = 40 / 1000 * 1.0Darcy * ones(4, 8)

# perm3[30:37, 33:36] = 40 / 1000 * 1.0Darcy * ones(8, 4)
# perm3[90:97, 33:36] = 40 / 1000 * 1.0Darcy * ones(8, 4)

# perm3[45:50, 43:48] = 40 / 1000 * 1.0Darcy * ones(6, 6)
# perm3[75:80, 43:48] = 40 / 1000 * 1.0Darcy * ones(6, 6)


# poro3 = ones(100, 50) * 0.27
# poro3[:, 1:2] .*= 0.0
# poro3[:, 50] .*= 0.02/0.27
# poro3[:, 12] .*= 0.11/0.27
# poro3[:, 22] .*= 0.11/0.27
# poro3[:, 32] .*= 0.11/0.27
# poro3[:, 42] .*= 0.11/0.27

# poro3[40:45, 3:8] = 0.11 * ones(6, 6)
# poro3[55:60, 3:8] = 0.11 * ones(6, 6)
# poro3[70:75, 3:8] = 0.11 * ones(6, 6)
# poro3[80:85, 3:8] = 0.11 * ones(6, 6)

# poro3[50:53, 13:16] = 0.11 * ones(4, 4)
# poro3[60:63, 13:16] = 0.11 * ones(4, 4)

# poro3[20:23, 23:30] = 0.11 * ones(4, 8)
# poro3[70:73, 23:30] = 0.11 * ones(4, 8)

# poro3[30:37, 33:36] = 0.11 * ones(8, 4)
# poro3[90:97, 33:36] = 0.11 * ones(8, 4)

# poro3[45:50, 43:48] = 0.11 * ones(6, 6)
# poro3[75:80, 43:48] = 0.11 * ones(6, 6)


# domain = reservoir_domain(mesh, permeability = vec(perm3), porosity = vec(poro3),temperature = convert_to_si(30.0, :Celsius))
# Injector = setup_well(domain, (65, 1, 43), name = :Injector)
# model, parameters = setup_reservoir_model(domain, :co2brine, wells = Injector);


# #plot the model
# plot_reservoir(model)


# save the model for future plotting purpose
save_object("scripts/3D_" * string(sp) * "/model" * ".jld2", model)



# setup schedule
nstep = 25
nstep_shut = 25
dt_inject = fill(365.0day, nstep)
pv = pore_volume(model, parameters)

extracted = [mesh.node_points[(i - 1) * 31 * 30 + 1 : i * 31 * 30] for i in 1:10]
depth_arr = [extracted[i][j][3] for i in 1:10 for j in 1:31*30] 


hydrau = depth_arr * 1000 * 9.807 

threshold = 1.5

## set the bound for the fracture pressure 

# default hydraulic bound
P_bound = hydrau .+ threshold * 10^6 

BHP_bound = maximum(P_bound)

# real bound, do extrapolation
# 3566 psi at 3688 ft
# 4100 psi at 5283 ft
# 6106 psi at 6350 ft
# 8526 psi at 8969 ft 

# 1 psi to 6894.757 Pa

# 1 feet to 0.3048 meter

# accepted fracture pressure raio
using Polynomials
ratio = 0.9

x = [3688, 5283, 6350, 8969] * 0.3048
y = [3566, 4100, 6106, 8526] * ratio * 6894.757

p = fit(x, y, 2)
 
# plot the fitted curve
x_fit = range(minimum(x), stop=maximum(x)+1, length=100)
y_fit = p.(x_fit)
lines(x_fit, y_fit)

# set the bound by the fitted curve
P_bound = p.(depth_arr)

BHP_bound = P_bound[Injector_cor[1] * Injector_cor[2] * Injector_cor[3]]


## Define the objective function with log barrier method, 
## input variable here is the injection rate



function myobjective(inj_rate)

    ## try to maximize the injection rate here
    ## inj_rate = 0.0075*sum(pv)/sum(dt_inject) 

    rate_target = TotalRateTarget(inj_rate)
    I_ctrl = InjectorControl(rate_target, [0.0, 1.0], density = 900.0)
    
    controls = Dict(:Injector => I_ctrl)
    forces_inject = setup_reservoir_forces(model, control = controls)

    forces_shut = setup_reservoir_forces(model)
    dt_shut = fill(365.0day, nstep_shut);   

    dt = vcat(dt_inject, dt_shut)
    forces = vcat(fill(forces_inject, nstep), fill(forces_shut, nstep_shut));

    # set up initial state
    state0 = setup_reservoir_state(model, Pressure = 200bar, OverallMoleFractions = [1.0, 0.0])

    # simulate the schedule
    wd, states, t = simulate_reservoir(state0, model, dt,
        parameters = parameters,
        forces = forces,
        max_timestep = 30day
    )   

    regu_term = 1e11

    obj = 0

    obj -= inj_rate * t[end] * 900 / regu_term

    for i in 1:50
        P = states[i][:Pressure] 

        # P_bound

        BHP = wd[:Injector][:bhp][i]

        # BHP_bound

        if any(x -> x < 0, P_bound - P) || any(x -> x < 0, BHP_bound - BHP) 
            obj += Base.Inf
        else
            obj -= (sum(log.(P_bound - P)) + sum(log.(BHP_bound - BHP))) * t[1] / (1e4 * regu_term)
        end

    end

    return obj, states

end


## The function to calculate the gradient of the objective function w.r.t the injection rate.
## It uses finite difference method with the h to be delta_q.
function gradient_wrt_inj(q)
    delta = 10^-6
    
    obj_plus, states = myobjective(q+delta)
    obj_minus, states = myobjective(q-delta)

    grad = (obj_plus - obj_minus) / (2 * delta)
    
    return grad
end


## do the optimization loop

## Projection operator for bound constraints
proj(x) = max.(x, 0)
ls = BackTracking(order=3, iterations=20)

niterations = 20

step_arr = zeros(niterations)

ex_step_size = 0.01

# set the initial injection rate
inj_rate = 0.01

inj_arr = zeros(Float64, niterations+1)
inj_arr[1] = inj_rate

# set the initial objective 
obj, _ =  myobjective(inj_rate)

# set the initial p, the gradient descent direction
grad = gradient_wrt_inj(inj_rate)
p = -grad/norm(grad, Inf)

obj_arr = zeros(Float64, niterations+1)



## Main loop
for j=1:niterations

    ## linesearch
    function θ(α)
        misfit, _ = myobjective(proj(inj_rate + α * p))
        @show α, misfit
        return misfit
    end

    step, obj = ls(θ, ex_step_size, obj, dot(grad, p))

    ex_step_size = step

    step_arr[j] = step
    inj_rate = proj(inj_rate + step * p)
    inj_arr[j+1] = inj_rate

    obj, states = myobjective(inj_rate)

    println("Iteration no: ",j,"; function value: ",obj) 

    # grad = gradient_wrt_inj(irateS, delta_irate, K_test)
    # p = -grad/norm(grad, Inf)

    # p = 1
    
    obj_arr[j+1] = obj
    
    save_object("scripts/3D_" * string(sp) * "/states" * "_iter" * string(j) * ".jld2" , states)

    # The accuracy of the solver is set to be 98% 
    
    if step < inj_rate * 0.02
        break
    end
    
end


save_object("scripts/3D_" * string(sp) * "/obj" * ".jld2", obj_arr)
save_object("scripts/3D_" * string(sp) * "/inj" * ".jld2", inj_arr)
save_object("scripts/3D_" * string(sp) * "/step" * ".jld2", step_arr)




## GLMakie plotting

# # plot the density of the brine
# using GLMakie
# function plot_co2!(fig, ix, x, title = "")
#     ax = Axis3(fig[ix, 1],
#         zreversed = true,
#         azimuth = -0.51π,
#         elevation = 0.05,
#         aspect = (1.0, 1.0, 0.3),
#         title = title)
#     plt = plot_cell_data!(ax, mesh, x, colormap = :seaborn_icefire_gradient)
#     Colorbar(fig[ix, 2], plt)
# end
# fig = Figure(size = (900, 1200))
# for (i, step) in enumerate([1, 5, nstep, nstep+nstep_shut])
#     plot_co2!(fig, i, states[step][:PhaseMassDensities][1, :], "Brine density report step $step/$(nstep+nstep_shut)")
# end
# save("s0.png", fig)
# fig


# plot the result in the interative viewer
# plot_reservoir(model, states)