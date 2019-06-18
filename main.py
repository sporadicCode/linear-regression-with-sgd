import numpy as np


def normalize(X):
    return X / np.max(np.abs(X), axis=0)


def sse_loss(w, X, y):
    yhat = w[0] + (w[1:] * X).sum(axis=1)
    err = y - yhat
    return np.square(err).sum()


# Linear regression with multiple variables, using stochastic gradient descent
def fit_sgd(X, y, n_iter=1000000, alpha=0.5) -> list:
    # alpha = 0.001...0.5
    rows_number = X.shape[0]
    weights_vector = np.zeros((5))
    intercept = 0
    sse_loss_memory = np.Inf
    for i in range(n_iter):

        # monitoring SSE loss
        if i % 20000 == 0 and i >= 20000:
            temp_arr = []
            temp_arr.append(intercept)
            for l in range(len(weights_vector)):
                temp_arr.append(weights_vector[l])
            temp_loss = sse_loss(temp_arr, X, y)
            if temp_loss > sse_loss_memory:
                break
            else:
                sse_loss_memory = temp_loss

        random_index = np.random.choice(rows_number)

        rand_x_sample = X[random_index]  # 5-long array
        rand_y_sample = y[random_index]  # 0/1

        # estimate = w0 + w1 * X1 + ... + wn * Xn 
        estimate = 0
        for j in range(len(rand_x_sample)):
            estimate += (rand_x_sample[j] * weights_vector[j])
        estimate += intercept

        err = rand_y_sample - estimate
        # print(f"Estimate: {estimate}, Actual: {rand_y_sample}, Error: {err}")

        intercept += alpha * err  # a.k.a. w0

        for k in range(len(weights_vector)):
            weights_vector[k] += (alpha * err * rand_x_sample[k])

    tmp_arr = []
    tmp_arr.append(intercept)
    for l in range(len(weights_vector)):
        tmp_arr.append(weights_vector[l])
    return tmp_arr


def test_model(w, X, y, threshold=0.5) -> float:
    yhat = w[0] + (w[1:] * X).sum(axis=1)      # value from linear model
    ypred = (yhat > threshold).astype(int)     # prediction
    ytrue = (ypred == y).astype(int)           # compare to true value
    return ytrue.sum() / y.shape[0]            # true count / total count


if __name__ == '__main__':

    train = np.loadtxt("training_data.csv", delimiter=",")
    test = np.loadtxt("test_data.csv", delimiter=",")

    X = normalize(train[:,:5])  # columns 0..4
    y = normalize(train[:,5])   # column 5  (classification: 0/1)

    X_test = normalize(test[:,:5])
    y_test = normalize(test[:,5])

    print(test_model(fit_sgd(X, y), X_test, y_test))
