import numpy as np
cimport numpy as np

LONG = np.int64
DBL = np.float64
INT = np.int32
ctypedef np.int64_t LONG_t
ctypedef np.float64_t DBL_t
ctypedef np.int32_t INT_t

def rebin_float(nb, ob, np.ndarray[DBL_t] ov, method, n):
    cdef np.ndarray[DBL_t] nv = np.zeros(n, dtype=DBL)
    return __rebin(nb, ob, ov, nv, method, n)

def rebin_long(nb, ob, np.ndarray[LONG_t] ov, method, n):
    cdef np.ndarray[LONG_t] nv = np.zeros(n, dtype=LONG)
    return __rebin(nb, ob, ov, nv, method, n)

def rebin_int(nb, ob, np.ndarray[INT_t] ov, method, n):
    cdef np.ndarray[INT_t] nv = np.zeros(n, dtype=INT)
    return __rebin(nb, ob, ov, nv, method, n)


def __rebin(nb, ob, ov, nv, method, n):
    if np.any(np.diff(nb) <= 0) or np.any(np.diff(ob) <= 0):
        raise ValueError('No zero or negative length bins allowed!')
    if (nb[0] < ob[0]) or (nb[-1] > ob[-1]):
        raise ValueError('New bins cannot extend beyond old bins.')

    # average and sum are very similar, so I will make avg work by using sum
    if method == 'avg':
        od = np.diff(ob)
        nd = np.diff(nb)
        sums = __rebin(nb, ob, ov*od, nv, 'sum', n)
        return sums/nd

    cdef np.ndarray[LONG_t] binmap = np.searchsorted(ob, nb, 'left')

    cdef size_t k, i, i0, i1

    #guess for speed I won't make it check what method to use in each loop iteration
    if method == 'sum':
        for k in range(n):
            i0 = binmap[k]
            i1 = binmap[k+1]
            if i0 == i1:
                nv[k] = (nb[k+1] - nb[k]) / (ob[i0] - ob[i0-1]) * ov[i0-1]
            else:
                left, right, mid = 0.0, 0.0, 0.0
                if nb[k] != ob[i0]:
                    left = (ob[i0] - nb[k]) / (ob[i0] - ob[i0-1]) * ov[i0-1]
                if nb[k+1] != ob[i1]:
                    right = (nb[k+1] - ob[i1-1]) / (ob[i1] - ob[i1-1]) * ov[i1-1]
                    i1 -= 1
                mid = 0.0
                for i in range(i0, i1): mid += ov[i]
                nv[k] = left + mid + right
    elif method == 'or':
        for k in range(n):
            i0 = binmap[k]
            i1 = binmap[k+1]
            if nb[k] != ob[i0] and i0 > 0:
                i0 = i0 - 1

            for i in range(i0, i1):
                nv[k] = nv[k] | ov[i]
    elif method == 'min':
        for k in range(n):
            i0 = binmap[k]
            i1 = binmap[k+1]
            if nb[k] != ob[i0] and i0 > 0:
                i0 = i0 - 1

            nv[k] = ov[i0]
            for i in range(i0+1, i1):
                if ov[i] < nv[k]:
                    nv[k] = ov[i]
    elif method == 'max':
        for k in range(n):
            i0 = binmap[k]
            i1 = binmap[k+1]
            if nb[k] != ob[i0] and i0 > 0:
                i0 = i0 - 1

            nv[k] = ov[i0]
            for i in range(i0+1, i1):
                if ov[i] > nv[k]:
                    nv[k] = ov[i]

    return nv


def rebin_rows(np.ndarray[DBL_t] nb, np.ndarray[DBL_t] ob, np.ndarray[DBL_t, ndim=2] ov, method):
    if np.any(np.diff(nb) <= 0) or np.any(np.diff(ob) <= 0):
        raise ValueError('No zero or negative length bins allowed!')
    if (nb[0] < ob[0]) or (nb[-1] > ob[-1]):
        raise ValueError('New bins cannot extend beyond old bins.')

    # average and sum are very similar, so I will make avg work by using sum
    if method == 'avg':
        od = np.diff(ob)
        nd = np.diff(nb)
        sums = rebin_rows(nb, ob, ov*od[None,:], 'sum')
        return sums/nd[None,:]

    cdef size_t n = len(nb) - 1
    cdef size_t m = len(ov)

    cdef np.ndarray[DBL_t, ndim=2] nv = np.zeros([m,n], dtype=DBL)

    cdef np.ndarray[LONG_t] binmap = np.searchsorted(ob, nb, 'left')

    cdef size_t k, i, i0, i1
    cdef double fac, leftfac, rightfac, mid

    #guess for speed I won't make it check what method to use in each loop iteration
    if method == 'sum':
        for k in range(n):
            i0 = binmap[k]
            i1 = binmap[k+1]
            if i0 == i1:
                fac = (nb[k+1] - nb[k]) / (ob[i0] - ob[i0-1])
                for j in range(m):
                    nv[j,k] = fac * ov[j,i0-1]
            else:
                if nb[k] != ob[i0]:
                    leftfac = (ob[i0] - nb[k]) / (ob[i0] - ob[i0-1])
                    i_left = i0-1
                else:
                    leftfac = 0.0
                    i_left = 0 # using this prevents an attempt at accessing index -1 if i0=0
                if nb[k+1] != ob[i1]:
                    rightfac = (nb[k+1] - ob[i1-1]) / (ob[i1] - ob[i1-1])
                    i1 -= 1
                    i_right = i1
                else:
                    rightfac = 0.0
                    i_right = 0
                for j in range(m):
                    mid = 0.0
                    for i in range(i0, i1):
                        mid += ov[j,i]
                    nv[j,k] = leftfac*ov[j,i_left] + mid + rightfac*ov[j,i_right]

    return nv

# this is all just so I can handle different data types
def rebin(np.ndarray[DBL_t] nb, np.ndarray[DBL_t] ob, ov, method):
    n = len(nb) - 1
    if type(ov.item(0)) is float:
        return rebin_float(nb, ob, ov.astype('f8'), method, n)
    elif type(ov.item(0)) is int:
        return rebin_int(nb, ob, ov.astype('i4'), method, n)
    elif type(ov.item(0)) is long:
        return rebin_long(nb, ob, ov.astype('i8'), method, n)


def bin(np.ndarray[DBL_t] b, np.ndarray[DBL_t] x, np.ndarray[DBL_t] y, method):

    if np.any(np.diff(b) <= 0):
        raise ValueError('No zero or negative length bins allowed!')
    if np.any(x[1:] <= x[:-1]):
        raise ValueError('x values must be monotonically increasing!')
    if (b[0] < x[0]) or (b[-1] > x[-1]):
        raise ValueError('Bins cannot extend beyond data.')

    # average and sum are very similar, so I will make avg work by using sum
    if method == 'avg':
        nd = np.diff(b)
        sums = bin(b, x, y, 'sum')
        return sums / nd

    n = len(b) - 1

    # xx and yy have the bin edge values inserted into the input values
    ii = np.searchsorted(x, b)
    cdef np.ndarray[LONG_t] binmap = ii + np.arange(len(ii), dtype=LONG)
    cdef np.ndarray[DBL_t] xx = np.insert(x, ii, b)
    yb = np.interp(b, x, y)
    cdef np.ndarray[DBL_t] yy = np.insert(y, ii, yb)

    cdef size_t i, k, i0, i1
    cdef np.ndarray[DBL_t] nv = np.zeros(n, dtype=DBL)

    if method == 'sum':
        for k in range(n):
            i0 = binmap[k]
            i1 = binmap[k+1]
            for i in range(i0, i1):
                nv[k] += 0.5*(yy[i+1] + yy[i])*(xx[i+1] - xx[i])

    return nv


def bin_rows(np.ndarray[DBL_t] b, np.ndarray[DBL_t] x, np.ndarray[DBL_t, ndim=2] y, method):

    if np.any(np.diff(b) <= 0):
        raise ValueError('No zero or negative length bins allowed!')
    if np.any(x[1:] <= x[:-1]):
        raise ValueError('x values must be monotonically increasing!')
    if (b[0] < x[0]) or (b[-1] > x[-1]):
        raise ValueError('Bins cannot extend beyond data.')

    # average and sum are very similar, so I will make avg work by using sum
    if method == 'avg':
        nd = np.diff(b)
        sums = bin_rows(b, x, y, 'sum')
        return sums / nd[None,:]

    m = len(y)
    n = len(b) - 1

    # xx and yy have the bin edge values inserted into the input values
    ii = np.searchsorted(x, b)
    cdef np.ndarray[LONG_t] binmap = ii + np.arange(len(ii), dtype=LONG)
    cdef np.ndarray[DBL_t] xx = np.insert(x, ii, b)
    yb = np.array([np.interp(b, x, _y) for _y in y])
    cdef np.ndarray[DBL_t, ndim=2] yy = np.insert(y, ii, yb, axis=1)

    cdef size_t i, j, k, i0, i1
    cdef np.ndarray[DBL_t, ndim=2] nv = np.zeros([m,n], dtype=DBL)

    if method == 'sum':
        for k in range(n):
            i0 = binmap[k]
            i1 = binmap[k+1]
            for i in range(i0, i1):
                dx2 = (xx[i+1] - xx[i])/2.0
                for j in range(m):
                    nv[j,k] += dx2*(yy[j,i+1] + yy[j,i])

    return nv