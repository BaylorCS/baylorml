LIBML
=====

Compilation Instructions
------------------------

Run:

         make

Some classes in this library use Eigen, a very fast C++ linear algebra library.
Eigen supports parallelization using OpenMP, and it will do so by default. If
you do not want Eigen to parallelize its operations, you should set the macro
EIGEN_DONT_PARALLELIZE before compiling this library.

