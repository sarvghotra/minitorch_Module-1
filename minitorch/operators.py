import math

## Task 0.1
## Mathematical operators


def mul(x, y):
    """
    Returns the multiplication of `x` and `y` :math:`f(x, y) = x * y`

    Args:
        x (scalar): A scalar of any type
        y (scalar): A scalar of any type

    Returns:
        x * y
    """
    return x * y


def id(x):
    """
    Identity function of `x` :math:`f(x) = x`

    Args:
        x (scalar): A scaler of any type

    Returns:
        x
    """
    return x


def add(x, y):
    """
    Returns sum of `x` and `y` :math:`f(x, y) = x + y`

    Args:
        x (scalar): A scalar of any type
        y (scalar): A scalar of any type

    Returns:
        x + y
    """
    return x + y


def neg(x):
    """
    Returns negatation of `x` :math:`f(x) = -x`

    Args:
        x (scalar): A scalar of any type

    Returns:
        x
    """
    return -1 * x


def lt(x, y):
    """
    Implements: :math:`f(x) =` 1.0 if x is less than y else 0.0

    Args:
        x (scalar): A scalar of any type
        y (scalar): A scalar of any type

    Returns:
        1.0 if x is less than y else 0.0
    """
    return 1.0 if x < y else 0.0


def eq(x, y):
    """
    Implements: ":math:`f(x) =` 1.0 if x is equal to y else 0.0"

    Args:
        x (scalar): A scalar of any type
        y (scalar): A scalar of any type

    Returns:
        1.0 if x is less than y else 0.0
    """
    return 1.0 if x == y else 0.0


def max(x, y):
    """
    Returns the max of `x` and `y` :math:`f(x) =` x if x is greater than y else y

    Args:
        x (scalar): A scalar of any type
        y (scalar): A scalar of any type

    Returns:
        x if x is greater than y else y
    """
    return x if x > y else y


def sigmoid(x):
    """
    Implements sigmoid function :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    Args:
        x (scalar): A scalar of any type

    Returns:
        Value of sigmoid function

    """
    if x >= 0:
        y = 1.0 / (1.0 + math.exp(-1.0 * x))
    else:
        y = math.exp(x) / (1.0 + math.exp(x))
    return y


def relu(x):
    """
    Implements Relu: :math:`f(x) =` x if x is greater than 0, else 0

    Args:
        x (scalar): A scalar of any type

    Returns:
        Output value of relu function
    """
    return x if x > 0 else type(x)(0.0)


def relu_back(x, y):
    """
    Implements :math:`f(x) =` y if x is greater than 0 else 0

    Args:
        x (scalar): A scalar of any type
        y (scalar): A scalar of any type

    Returns:
        y if x is greater than 0 else 0
    """
    return y if x > 0 else 0


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(a, b):
    return b / (a + EPS)


def inv(x):
    ":math:`f(x) = 1/x`"
    return 1.0 / x


def inv_back(a, b):
    return -(1.0 / a ** 2) * b


## Task 0.3
## Higher-order functions.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): process one value

    Returns:
        function : a function that takes a list and applies `fn` to each element
    """
    # TODO: Implement for Task 0.3.
    res = []
    def apply_fn(lst):
        for l in lst:
            res.append(fn(l))
        return res

    return apply_fn


def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) one each pair of elements.

    """
    res = []
    def apply_fn(lst1, lst2):
        assert len(lst1) == len(lst2)

        for l1, l2 in zip(lst1, lst2):
            res.append(l1 + l2)
        return res

    return apply_fn


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`

    """
    def apply_fn(ls):
        res = start
        for l in ls:
            res = fn(l, res)
        return res

    return apply_fn


def sum(ls):
    """
    Sum up a list using :func:`reduce` and :func:`add`.

    Args:
        ls (List): A list of numbers

    Returns:
        scalar : Sum of list `ls`
    """
    return reduce(add, 0.0)(ls)


def prod(ls):
    """
    Product of a list using :func:`reduce` and :func:`mul`.

    Args:
        ls (List): A list of numbers

    Return:
        scalar : Product of List `ls` elements
    """
    return reduce(mul, 1.0)(ls)
