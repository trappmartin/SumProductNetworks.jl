using SumProductNetworks

using MLDatasets

train_x, train_y = MNIST.traindata()
train_x = MNIST.convert2features(train_x)
train_1 = train_x[:, train_y .== 1]
train_2 = train_x[:, train_y .== 2]
train_3 = train_x[:, train_y .== 3]
train_4 = train_x[:, train_y .== 4]

spn = generate_spn(train_2, :learnspn)
updatescope!(spn)

println(logpdf(spn, train_1[:, 1]))
println(logpdf(spn, train_2[:, 1]))
println(logpdf(spn, train_3[:, 1]))
println(logpdf(spn, train_4[:, 1]))
