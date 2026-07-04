import numpy as np
import sklearn

################################
# Non Editable Region Starting #
################################


def my_map(X):
    ################################
    #  Non Editable Region Ending  #
    ################################
    original = X

    even = np.arange(0, 32, 2)
    odd = np.arange(1, 32, 2)
    X_even = X[:, even]
    X_odd = X[:, odd]

    quadratic = (
        X_even[:, :, None] * X_odd[:, None, :]
    )
    
    quadratic = quadratic.reshape(X.shape[0], -1)

    X_map = np.concatenate([original, quadratic], axis=1)
    return X_map


################################
# Non Editable Region Starting #
################################
def my_params(X_map, X_raw, y):
    ################################
    #  Non Editable Region Ending  #
    ################################
    my_params = {
        "tol": 1e-3,
        "C": 1.0,
        "random_state": 42,
        "max_iter": 50
    }
    return my_params
