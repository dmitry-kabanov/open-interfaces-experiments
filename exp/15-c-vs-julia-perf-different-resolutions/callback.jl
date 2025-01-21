module CallbackWrapper
export make_wrapper_for_c_callback

import SciMLBase

using OpenInterfaces: OIFArrayF64

function make_wrapper_over_oif_c_callback(fn_c::Ptr{Cvoid})::Function
    function wrapper(t, y, ydot, user_data)::Int
        if typeof(user_data) == SciMLBase.NullParameters
            user_data = C_NULL
        elseif user_data isa Tuple
            user_data = Ref(user_data)
        else
            error("Unsupported `user_data` type: $(typeof(user_data))")
        end

        oif_y = _oif_array_f64_pointer_from_array_f64(y)
        oif_ydot = _oif_array_f64_pointer_from_array_f64(ydot)

        @ccall $fn_c(t::Float64, oif_y::Ptr{OIFArrayF64}, oif_ydot::Ptr{OIFArrayF64}, user_data::Ptr{Cvoid})::Cint
        return 0
    end
    return wrapper
end

function make_wrapper_over_carray_c_callback(fn_c::Ptr{Cvoid})::Function
    function wrapper(t::Float64, y::Vector{Float64}, ydot::Vector{Float64}, user_data::Tuple)::Int
        if typeof(user_data) == SciMLBase.NullParameters
            user_data = C_NULL
        elseif user_data isa Tuple
            user_data = Ref(user_data)
        else
            error("Unsupported `user_data` type: $(typeof(user_data))")
        end
        c_y = Base.unsafe_convert(Ptr{Float64}, y)
        c_ydot = Base.unsafe_convert(Ptr{Float64}, ydot)
        N = length(y)

        @ccall $fn_c(t::Float64, c_y::Ptr{Float64}, c_ydot::Ptr{Float64}, user_data::Ptr{Cvoid}, N::Csize_t)::Cint
        return 0
    end
    return wrapper
end

function _oif_array_f64_pointer_from_array_f64(arr::AbstractArray{T, N}) where {T<:Float64, N}
    ndim = ndims(arr)
    dimensions = Base.unsafe_convert(Ptr{Clong}, collect(size(arr)))
    data = Base.unsafe_convert(Ptr{Float64}, arr)
    oif_arr = Ref(OIFArrayF64(ndim, dimensions, data))
    return oif_arr
end
end

