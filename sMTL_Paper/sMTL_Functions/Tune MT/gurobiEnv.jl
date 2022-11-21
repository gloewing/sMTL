using JuMP, Gurobi
# finds gurobi enviornment so we can pass this to other functions
# and use on cluster without always making a new gurobi enviornment

function gurobiEnv()
    # const GRB_ENV = Gurobi.Env();
    GRB_ENV = Gurobi.Env();
    return GRB_ENV

end
