# Plot the model and states varying along time for the results from 
# the control framework
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
using CSV
using DataFrames

# speficiy which sample and which optimization iteration
sp = 0
iter = 6

model = load_object("3D_" * string(sp) * "/model" * ".jld2")

states = load_object("3D_" * string(sp) * "/states" * "_iter" * string(iter) * ".jld2")

# plot the model
plot_reservoir(model)

# plot the result in the interative viewer
plot_reservoir(model, states)