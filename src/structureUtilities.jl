export generate_spn

"""
    generate_spn(X::Matrix, algo::Symbol; ...)

Generate an SPN structure using a structure learning algorithm.

Arguments:
* `X`: Data matrix.
* `algo`: Algorithm, one out of [:learnspn, :random].
"""
function generate_spn(X::Matrix, algo::Symbol; params...)

    if algo == :learnspn
        return learnspn(X; params...)
    elseif algo == :random
        return randomspn(X; params...)
    else
        @error("Unknown structure learning method: ", method)
    end
end

learnspn(X;params...) = @error("LearnSPN is currently not supported.")
randomspn(X;params...) = @error("Random structure learning is currently not supported.")
