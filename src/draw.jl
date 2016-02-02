function drawSPN(spn::SumNode; file="spn.svg")

   nodes = order(spn)

   nodeTypes = Vector{Symbol}(0)
   A = zeros(Base.length(nodes), Base.length(nodes))

   reverse!(nodes)

   for i in collect(1:Base.length(nodes))
      if isa(nodes[i], Node)
         for j in collect(1:Base.length(nodes))
            if nodes[j] in nodes[i].children
               A[i, j] = 1
            end
         end
         if isa(nodes[i], SumNode)
            push!(nodeTypes, :sum)
         else
            push!(nodeTypes, :product)
         end
      else
         push!(nodeTypes, :leaf)
      end
   end

   adj_list = Vector{Int}[]
   for i in 1:size(A,1)
       new_list = Int[]
       for j in 1:size(A,2)
           if A[i,j] != zero(eltype(A))
               push!(new_list,j)
           end
       end
       push!(adj_list, new_list)
   end

   layoutSPN(adj_list, nodeTypes, cycles=false, filename=file)

end

function drawSPN(spn::SPNStructure; file="spn.svg")

   nodeTypes = Vector{Symbol}(0)
   

   #layoutSPN(adj_list, nodeTypes, cycles=false, filename=file)

end

function layoutSPN{T}(adj_list::Vector{Vector{T}},
                        nodeTypes::Vector{Symbol};
                        filename    = "",

                        cycles      = true,
                        ordering    = :optimal,
                        coord       = :optimal,
                        xsep        = 3,
                        ysep        = 20,
                        scale       = 0.1,
                        labelpad    = 1.2,
                        background  = nothing)
    # Calculate the original number of vertices
    n = length(adj_list)

    # 2     Layering
    # 2.1   Assign a layer to each vertex
    layers = GraphLayout._layer_assmt_longestpath(adj_list)
    num_layers = maximum(layers)
    # 2.2  Create dummy vertices for long edges
    adj_list, layers = GraphLayout._layer_assmt_dummy(adj_list, layers)
    orig_n, n = n, length(adj_list)


    # 3     Vertex ordering [to reduce crossings]
    # 3.1   Build initial permutation vectors
    layer_verts = [L => Int[] for L in 1:num_layers]
    for i in 1:n
        push!(layer_verts[layers[i]], i)
    end
    # 3.2  Reorder permutations to reduce crossings
    if ordering == :barycentric
        layer_verts = GraphLayout._ordering_barycentric(adj_list, layers, layer_verts)
    elseif ordering == :optimal
        layer_verts = GraphLayout._ordering_ip(adj_list, layers, layer_verts)
    end


    # 4     Vertex coordinates [to straighten edges]
    # 4.1   Place y coordinates in layers
    locs_y = zeros(n)
    for L in 1:num_layers
        for (x,v) in enumerate(layer_verts[L])
            locs_y[v] = (L-1)*ysep
        end
    end
    # 4.2   Get widths of each label, if there are any
    widths  = ones(n) * 2; widths[orig_n+1:n]  = 0
    heights = ones(n) * 2; heights[orig_n+1:n] = 0

    # Note that we will convert these sizes into "absolute" units
    # and then work in these same units throughout. The font size used
    # here is just arbitrary, and unchanging. This hack arises because it
    # is meaningless to ask for the size of the font in "relative" units
    # but we don't want to collapse to absolute units until the end.
    #if length(labels) == orig_n
    # extents = text_extents("sans",10pt,labels...)
    #    for (i,(width,height)) in enumerate(extents)
    #        widths[i]  = width.value
    #        heights[i] = height.value
    #    end
    #end
    locs_x = GraphLayout._coord_ip(adj_list, layers, layer_verts, orig_n, widths, xsep)
    # 4.3   Summarize vertex info
    max_x, max_y = maximum(locs_x), maximum(locs_y)
    max_w, max_h = maximum(widths), maximum(heights)

    # 5     Draw the tree
    # 5.1   Create the vertices
    verts = [_tree_node(locs_x[i], locs_y[i], nodeTypes[i], widths[i], heights[i], max_x, max_y) for i in 1:orig_n]
    # 5.2   Create the arrows
    arrows = Any[]
    for L in 1:num_layers, i in layer_verts[L], j in adj_list[i]
        push!(arrows, _arrow_tree(
                locs_x[i], locs_y[i], i<=orig_n ? max_h : 0,
                locs_x[j], locs_y[j], j<=orig_n ? max_h : 0))
    end
    # 5.3   Assemble composition
    # We need to decide the true font size now. The logic is as follows:
    # - All the text has the same height (max_h)
    # - The image is max_h+max_y units tall
    # - The real image will be scale*max_y inches tall
    # - Therefore one unit = ...
    ratio = (scale*max_y)/(max_y+max_h)

    #
    #   inchs.
    # - Now we want to scale the text size, in inches, so that
    #   the height of the text (in inches) is approximately the height
    #   we already have in 'units'. We'll fudge font size = font height,
    #   then use the padding factor to scale the text. Bigger padding,
    #   smaller font relative to its box.
    fsize = (1.0/labelpad) * max_h * (scale*max_y)/(max_y+max_h) * inch
    # Determine the background, if we want one
    bg = background == nothing ? [] : [rectangle(), fill(background)]
    c = compose(
        context(units=UnitBox(-max_w/2,-max_h/2,max_x+max_w,max_y+max_h)),
        font("sans"), fontsize(fsize),
        bg..., verts..., arrows...
    )
    # 5.4   Draw it
    if filename != ""
        Compose.draw(SVG(filename, scale*max_x*inch, scale*max_y*inch), c)
    end
    return c
end

function _tree_node(x, y, nodeType::Symbol, width, height, max_x, max_y)
    if nodeType == :sum
        _tree_sum(x, y, width, height, max_x, max_y)
    elseif nodeType == :product
        _tree_product(x, y, width, height, max_x, max_y)
    else
        _tree_leaf(x, y, width, height, max_x, max_y)
    end
end

_tree_sum(x, y, width, height, max_x, max_y) = compose(
        context(x - width/2, y - height/2, width, height),
        (context(), circle(), stroke(colorant"black"), fill(nothing)),
        (context(), line([(0.5 * max_x, max_y), (0.5 * max_x, 0.0)]), stroke(colorant"black")),
        (context(), line([(max_x, 0.5 * max_y), (0.0, 0.5 * max_y)]), stroke(colorant"black"))
      )
#
_tree_product(x, y, width, height, max_x, max_y) = compose(
        context(x - width/2, y - height/2, width, height),
        (context(), circle(), stroke(colorant"black"), fill(nothing)),
        (context(), line([(0.09 * max_x, 0.91 * max_y), (0.91 * max_x, 0.09 * max_y)]), stroke(colorant"black")),
        (context(), line([(0.09 * max_x, 0.09 * max_y), (0.91 * max_x, 0.91 * max_y)]), stroke(colorant"black"))
    )

_tree_leaf(x, y, width, height, max_x, max_y) = compose(
        context(x - width/2, y - height/2, width, height),
        (context(), circle(), stroke(colorant"black"), fill(nothing)),
        (context(), curve((0.1 * max_x, 0.5 * max_y), (0.4 * max_x, 1.0 * max_y), (0.25 * max_x, 0.2 * max_y), (0.501 * max_x,0.2 * max_y), :curve), stroke(colorant"black")),
        (context(), curve((0.9 * max_x, 0.5 * max_y), (0.6 * max_x, 1.0 * max_y), (0.75 * max_x, 0.2 * max_y), (0.499 * max_x, 0.2 * max_y), :curve), stroke(colorant"black"))
	)

@doc """
    Creates an arrow between two rectangles in the tree
    Arguments:
    o_x, o_y, o_h   Origin x, y, and height
    d_x, d_y, d_h   Destination x, y, and height
""" ->
function _arrow_tree(o_x, o_y, o_h, d_x, d_y, d_h)
    x1, y1 = o_x, o_y + o_h/2
    x2, y2 = d_x, d_y - d_h/2
    Δx, Δy = x2 - x1, y2 - y1
    θ = atan2(Δy, Δx)
    # Put an arrow head only if destination isn't dummy
    head = d_h != 0 ? _arrow_heads(θ, x2, y2, 2) : []
    compose(context(), stroke("black"),
        line([(x1,y1),(x2,y2)]), head...)
end



@doc """
    Creates an arrow head given the angle of the arrow and its destination.
    Arguments:
    θ               Angle of arrow (radians)
    dest_x, dest_y  End of arrow
    λ               Length of arrow head tips
    ϕ               Angle of arrow head tips relative to angle of arrow
""" ->
_arrow_heads(θ, dest_x, dest_y, λ, ϕ=0.125π) = [ line([
    (dest_x - λ*cos(θ+ϕ), dest_y - λ*sin(θ+ϕ)),
    (dest_x, dest_y),
    (dest_x - λ*cos(θ-ϕ), dest_y - λ*sin(θ-ϕ))
]) ]
