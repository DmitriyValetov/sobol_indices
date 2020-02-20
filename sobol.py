import numpy as np


def analyze(x, y, n=5, n_boot=50, print_to_console=False):
    """
    Calculate first order indices.
    x (N, dim) np.array - input samples
    y (N,) np.array - output value
    n int - number of sections for dispersion analysis
    n_boot int - bootstrap for dispersion accumulation
    """
    y = (y - y.mean()) / y.std()
    D1 = {i: 0 for i in range(X.shape[1])}
    n_size = x.shape[0]//n

    for i in range(x.shape[1]):
        zipped = np.array(sorted(np.concatenate([x, np.expand_dims(y, 1)], axis=1), key=lambda x: x[i])) # align for slicing
        delimeter = 0
        for j in range(n):
            for _ in range(n_boot):
                group = zipped[n_size*j:n_size*(j+1), :]
                np.random.shuffle(group)
                half_1_i = np.random.choice(group.shape[0], group.shape[0]//2, replace=False)
                half_1 = group[half_1_i]
                half_2_i = list(set(range(group.shape[0])) - set(half_1_i))[:group.shape[0]//2]
                half_2 = group[half_2_i]
                D1[i] += np.sum(half_1[:, -1]*half_2[:, -1])
                delimeter += half_1.shape[0]

        D1[i] = D1[i]/delimeter
        if print_to_console:
            print(f'S1 x{i+1}: {D1[i]}')
    
    if print_to_console:
        print(f"First order dispersion: {sum([D1[i] for i in D1])}") # if all linear this will be near 1.0
        print(f"Higher order dispersion: {1-sum([D1[i] for i in D1])}")


if __name__ == "__main__":
    np.random.seed(42)

    def make_foo(a):
        def g(x, a):
            return (abs(4*x-2)+a)/(1+a)
        
        def foo(x):
            return np.prod([g(x[i], a[i]) for i in range(len(x))])
        
        return foo

    a = [0,1,9,99] # the more - the less will be sensitivity
    foo = make_foo(a)
    problem = {
        'num_vars': 4,
        'names': ['x1', 'x2', 'x3', 'x4'],
        'bounds': [[0.0, 1.0]]*4
    }

    X = np.random.uniform([0.0]*4, [1.0]*4, (1000, 4) )
    Y = np.array([foo(x) for x in X])
    analyze(X, Y, n=10, n_boot=50, print_to_console=True)