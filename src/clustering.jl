using Clustering, Distances

function stnll(x, m, a, c, B, D)
	# Compute Student-t negative log likelihood (Appendix A, eqn. (20))

	mu = m'
	nu = a-D+1.
	Lambda = c * nu / (c+1.) * B

	S = ((x-mu)' * Lambda) * (x-mu)

	logdetL = logdet(Lambda)
	return nu+D/2.*log.(1.+S/nu) - 0.5*logdetL+lgamma(nu/2.) - lgamma((nu+D)/2.)+D/2.*log(nu*pi)
end

function nwupd(Nki, xki, m0, a0, c0, B0)
	# Update Normal-Wishart hyper parameters (Appendix A, eqns. (18-19))

	xmki = mean(xki, 1)
	xmcki = xki .- xmki
	Ski = xmcki' * xmcki   
	cki = c0+Nki
	mki = (c0*m0+Nki*xmki)/cki
	xm0cki = xmki-m0
	Bki = inv(inv(B0)+Ski+c0*Nki/cki* (xm0cki' * xm0cki))
	aki = a0+Nki
	return (mki, aki, cki, Bki)
end

# MAP - DP code based on:
# Yordan P. Raykov, Alexis Boukouvalas, Fahd Baig, Max A. Little (2016)
# "What to do when K-means clustering fails: a simple yet principled alternative algorithm",
# PLoS One, (11)9:e0162259
#
# N0 ... concentration parameter
function mapDP(X, N0, m0, a0, c0, B0; ϵ = 1e-10, maxIter = 100)

	(N, D) = size(X)

	K = 2

	(c1, c2) = kmpp(X', 2)
	Dist = pairwise(Euclidean(), X')[[c1, c2],:]
	#z = ones(Int, N) # initial assignments
	z = Int[indmin(Dist[:,i]) for i in 1:N]

	Enew = Inf
	dE = Inf
	ic = 0 #iteration count
	E = []

	# pre-compute student-t NLL for new cluster
	newNLL = map(i -> stnll(X[i,:], m0, a0, c0, B0, D)[1], 1:N)

	while (abs(dE) > ϵ) & (ic < maxIter)
		Eold = Enew
		dik = ones(N, 1) * Inf

		for i in 1:N
			dk = ones(K+1, 1) * Inf
			f = Vector{Float64}(K+1)
			Nki = ones(Int, K)
			xi = X[i,:]

			for k in 1:K
				zki = (z .== k)
				zki[i] = false
				Nki[k] = sum(zki)

				# updates not necessary if Nki == 0
				if Nki[k] == 0
					continue
				end

				# udpate
				(mki, aki, cki, Bki) = nwupd(Nki[k], X[zki, :], m0, a0, c0, B0)

				# compute student-t NLL
				p = stnll(xi, mki, aki, cki, Bki, D)
				dk[k] = p[1]

				# avoid reinforcement effet at initialization
				#if ic == 0
				#    Nki[1] = 1
				#end

				f[k] = dk[k] - log(Nki[k])
			end

			# student-t NLL for new cluster
			dk[K+1] = newNLL[i]
			f[K+1] = dk[K+1] - log(N0)

			# compute MAP assignment
			z[i] = indmin(f)
			dik[i] = f[z[i]]

			# create new cluster if required
			if z[i] > K
				K += 1
			end
		end

		# remove empty clusters and reassign
		Knz = 1
		for k in 1:K
			i = (z .== k)
			Nk = sum(i)
			if Nk > 0
				z[i] = Knz
				Knz += 1
			end
		end

		K = Knz - 1
		Nk = counts(z, 1:K)

		Enew = sum(dik) - K * log(N0) - sum(lgamma.(Nk))
		@assert !isinf(Enew)
		dE = Eold - Enew
		ic += 1
		push!(E, Enew)
	end

	return z
end


