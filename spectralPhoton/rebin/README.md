crebin
======

Code to rapidly and accurately rebin binned data (such as spectra) with flexibility in how values are combined in the binning. 

Note that I know next to nothing about Cython, in which this is written. Basically, I needed a fast rebinning code for Python and couldn't find one, so I flailed around with Cython until I got something that worked and then never touched it again until just now when I went to share it on Github. 

Normal rebinning takes the average of the values of the bins, weighted by the width of each bin. For example, if a new bin covers 40% of bin a, all of bin b, and 70% of bin c, then the value in that new bin is

(0.4*a + 1.0*b + 0.7*c)/(0.4*w_a + 1.0*w_b + 0.7*w_c)

where a,b, and c are the values in the bins and w_a, w_b, and w_c are the bin widths.

Sometimes, however, you might desire other types of binning. For example, if you are tracking data quality flags, then you will want to "or" the bins. That is, if two bins are combined and one has a data quality flag, you want the new bin to have that flag as well.

As of 2017/05/10 `crebin` supports avg (average, most common choice), sum, and, or, min, and max methods for rebinning. 

Import crebin like

```
from crebin import rebin
```

(I'm not sure why I made a crebin module with rebin inside, but whatever.)

Example:

```
oldbins = np.arange(6, dtype='f8')
newbins = np.array([0, 0.5, 2.5, 3.5, 4.0])
y = np.array([0, 1., 2, 1, 0])
rebin.rebin(newbins, oldbins, y, 'avg')
```

This supports numpy data types of int32, int64, or float64 for y. The bin edges must be float64 data type.

where newbins and oldbins define the bin edges (no gaps!) and y the values in each oldbin, so len(oldbins) = len(old_y) + 1. 

I added a second function to rebin ordinate data -- i.e. to take a series of points and compute the integral within a series of bins. As of 2017/05/10 this function only supports avg and sum methods. All arguments must be double precision arrays.

Example:

```
x = np.arange(5, dtype='f8')
y = np.array([0, 1., 2, 1, 0])
b = np.array([0, 1.0, 2.0, 3.5, 4.0])
rebin.bin(b, x, y, 'avg')
```

Aaaand I added a third function for fast rebinning of multiple data vectors that use the same binning. Currently it only supports data types of float64 for y. DISCLAIMER: I haven't tested this to see if it is actually faster because I realized what I really needed was a bin_rows function rather than rebin_rows.

```
oldbins = np.arange(6, dtype='f8')
newbins = np.array([0, 0.5, 2.5, 3.5, 4.0])
y = np.array([[0, 1., 2, 1, 0]]*4)
rebin.rebin_rows(newbins, oldbins, y, 'avg')
```
And now a fourth -- bin_rows. It's about 2x faster than looping bin. 

```
x = np.arange(5, dtype='f8')
y = np.array([[0, 1., 2, 1, 0]]*4)
b = np.array([0, 1.0, 2.0, 3.5, 4.0])
rebin.bin_rows(b, x, y, 'avg')
```

