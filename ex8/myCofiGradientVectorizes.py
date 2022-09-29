def cofi_gradient(params, Y, R, num_users, num_movies, num_features, _lambda):
    # unpack values
    X = params[:num_movies*num_features].reshape((num_movies, num_features))
    Theta = params[num_movies*num_features:].reshape((num_users, num_features))

    X_grad = np.zeros_like(X)
    Theta_grad = np.zeros_like(Theta)

    for i in range(len(X_grad)): # loop over rows

        # find users which rated the movie X[i]
        # X[np.where(idx == centroid)]
        idx = np.where(R[i] == 1)

        Theta_temp = Theta[idx]
        Y_temp = Y[i, idx]

        # nesto puta 1, nesto puta 100
        X_grad[i] = (Theta_temp @ X[i].T - Y_temp) @ Theta_temp + _lambda * X[i]

        # not vectorized version
        # for k in range(len(X_grad[0])): # loop over columns
            #X_grad[i][k] = np.sum((R[i] * (Theta @ X[i].T) - Y[i]) * Theta[:, k])

    for j in range(len(Theta_grad)): # loop over rows

        # find movies which were rated by user Theta[j]
        idx = np.where(R[:, j] == 1)

        X_temp = X[idx]
        Y_temp = Y[idx, j]

        Theta_grad[j] = (Theta[j] @ X_temp.T - Y_temp) @ X_temp + _lambda * Theta[j]

        # not vectorized version
        #for k in range(len(Theta_grad[0])): # loop over columns
            #Theta_grad[j][k] = np.sum((R[:, j] * (Theta[j] @ X.T) - Y[:, j]) * X[:, k])

    gradients_squashed = np.zeros(num_movies * num_features + num_users * num_features)
    gradients_squashed[:num_movies * num_features] = X_grad.flatten()
    gradients_squashed[num_movies * num_features:] = Theta_grad.flatten()

    return gradients_squashed

    # X_grad = np.sum((R * (X @ Theta.T) - Y) @ Theta.T)

    #return X_grad, Theta_grad
