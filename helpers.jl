module Helpers

export getOutdir

function getOutdir(script_filename::String = "")::String
    if isempty(script_filename)
        error("Invoke this function with the argument @__FILE__")
    end
    prefix = pwd() * "/"
    curdir_simplified = chopprefix(dirname(script_filename), prefix)
    outdir = curdir_simplified * "/_output"
    return outdir
end
end
