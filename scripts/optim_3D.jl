## Maximize injection rate within fracture pressure bound with gradient-based 
## backtracking line search optimization solver

import Pkg; Pkg.add("DrWatson")

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

nx = 100
nz = 50
Darcy, bar, kg, meter, day = si_units(:darcy, :bar, :kilogram, :meter, :day)


# Set up a 2D acquifer model
cart_dims = (nx, 1, nz)
physical_dims = (1000.0, 100.0, 50.0)
mesh = UnstructuredMesh(CartesianMesh(cart_dims, physical_dims))


points = mesh.node_points
for (i, pt) in enumerate(points)
    x, y, z = pt
    x_u = 2*π*x/1000.0
    w = 0.2
    dz = 0.05*x + w*(30*sin(2.0*x_u) + 20*sin(5.0*x_u) + 10*sin(10.0*x_u) + 5*sin(25.0*x_u))
    points[i] = pt + [0, 0, dz]
end


## 0.orginal universal permeability

# setup the simulation model
domain = reservoir_domain(mesh, permeability = 1.0Darcy, porosity = 0.3, temperature = convert_to_si(30.0, :Celsius))
Injector = setup_well(domain, (65, 1, 1), name = :Injector)
model, parameters = setup_reservoir_model(domain, :co2brine, wells = Injector);



# plot the model
# plot_reservoir(model)


# setup schedule
nstep = 25
nstep_shut = 25
dt_inject = fill(365.0day, nstep)
pv = pore_volume(model, parameters)

extracted = [mesh.node_points[(i - 1) * 31 * 30 + 1 : i * 31 * 30] for i in 1:10]
depth_arr = [extracted[i][j][3] for i in 1:10 for j in 1:31*30] 


hydrau = depth_arr * 1000 * 9.807 

threshold = 1.5

# set the bound for the fracture pressure 
P_bound = hydrau + threshold * 10^6 


BHP_bound = maximum(P_bound)


# 1 psi to 6894.757 Pa

# 1 feet to 0.3048 meter


## Define the objective function with log barrier method, 
## input variable here is the injection rate
function myobjective(inj_rate)

    ## try to maximize the injection rate here
    # inj_rate = 0.0075*sum(pv)/sum(dt_inject)



    rate_target = TotalRateTarget(inj_rate)
    I_ctrl = InjectorControl(rate_target, [0.0, 1.0],
        density = 900.0,
    )


    controls = Dict(:Injector => I_ctrl)
    forces_inject = setup_reservoir_forces(model, control = controls)

    forces_shut = setup_reservoir_forces(model)
    dt_shut = fill(365.0day, nstep_shut);

    dt = vcat(dt_inject, dt_shut)
    forces = vcat(fill(forces_inject, nstep), fill(forces_shut, nstep_shut));

    # set up initial state
    state0 = setup_reservoir_state(model,
        Pressure = 200bar,
        OverallMoleFractions = [1.0, 0.0],
    )

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
        P = states[i][:Presure] 

        P_bound

        BHP = wd[:Injector][:bhp][i]

        BHP_bound


        if any(x->x<0, P - P_bound) || any(x->x<0, BHP - BHP_bound) 
            obj += Base.Inf
        else
            obj -= (sum(log.(P - P_bound)) + sum(log.(BHP - BHP_bound))) * t[1] / (1e4 * regu_term)
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
ls = BackTracking(order=3, iterations=10)

niterations = 10

step_arr = zeros(niterations)

ex_step_size = 0.01

# set the initial injection rate
inj_rate = 0.03

inj_arr = zeros(Float64, niterations+1)
inj_arr[1, :] = inj_rate

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

    step, fval = ls(θ, ex_step_size, fval, dot(grad, p))

    ex_step_size = step

    step_arr[j] = step
    inj_rate = proj(inj_rate + step * p)
    inj_arr[j+1, :] = inj_rate

    obj, states = myobjective(inj_rate)

    println("Iteration no: ",j,"; function value: ",fval) 

    # grad = gradient_wrt_inj(irateS, delta_irate, K_test)
    # p = -grad/norm(grad, Inf)

    # p = 1

    obj_arr[j+1] = obj


    save_object("3D_0/states" * "_iter" * string(j) * ".jld2" , states)


    if step < inj_rate * 0.02
        break
    end

end


save_object("3D_0/" * ".jld2", obj_arr)
save_object("3D_0/" * ".jld2", inj_arr)
save_object("3D_0/" * ".jld2", step_array)





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