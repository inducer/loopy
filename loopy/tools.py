from __future__ import division




def generate_unique_possibilities(prefix):
    yield prefix

    try_num = 0
    while True:
        yield "%s_%d" % (prefix, try_num)
        try_num += 1

