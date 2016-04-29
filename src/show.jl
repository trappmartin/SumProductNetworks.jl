function Base.show(io::IO, m::ProductLayer)
  print(io, "ProductLayer => [#children: $(length(m.children))]")
end

function Base.show(io::IO, m::SumLayer)
  print(io, "SumLayer => [#nodes: $(length(m))]")
end
