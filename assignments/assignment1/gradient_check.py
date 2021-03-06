import numpy as np


def check_gradient(f, x, delta=1e-5, tol=1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''

    assert isinstance(x, np.ndarray)
    assert x.dtype == float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)

    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        numeric_grad_at_ix = 0

        if len(x.flatten()) == 1:
          numeric_grad_at_ix = (f(x + delta)[0] - f(x - delta)[0]) / (2 * delta)
        else:
          delta_array = np.zeros(x.shape)
          delta_array[ix] = delta
          x_delta_pos = x.copy()
          x_delta_pos += delta_array
          x_delta_neg = x.copy()
          x_delta_neg -= delta_array

          numeric_grad_at_ix = (f(x_delta_pos)[0] - f(x_delta_neg)[0]) / (2 * delta)

        print(f'analytic grad: {analytic_grad_at_ix}, numeric grad: {numeric_grad_at_ix}\n')

        # TODO compute value of numeric gradient of f to idx
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!\n")
    return True
        
