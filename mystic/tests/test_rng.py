from mystic import pseudorandom, seeded, unseeded


if __name__ == '__main__':

    def call_once(x):
        import numpy as np
        return x + np.random.random()

    # call the function
    try:
        #import FAKE
        from pathos.pools import ProcessPool
        p = ProcessPool()
        map = p.map
        PARALLEL = True
    except ImportError:
        PARALLEL = False

    func = seeded(call_once)
    y1 = list(map(func, [1,1]))
    y2 = list(map(func, [1,1]))
    assert y1 == y2
    assert y1[0] == y1[1]

    func = unseeded(call_once)
    y1 = list(map(func, [1,1]))
    y2 = list(map(func, [1,1]))
    assert y1 != y2
    assert y1[0] != y1[1]

    func = pseudorandom(call_once)
    y1 = list(map(func, [1,1]))
    y2 = list(map(func, [1,1]))
    assert y1 != y2 if PARALLEL else y1 == y2
    assert y1[0] != y1[1] if PARALLEL else y1[0] == y1[1]

    if PARALLEL:
        p.close(); p.join()
