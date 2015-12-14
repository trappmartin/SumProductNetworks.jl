# load data

using UCIMLRepo

df = ucirepodata("ionosphere")

X = float(convert(Array, df)[1:34, :])

# define intital model

 
