
Optimizing custom functions
---------------------------

Custom benchmark class
^^^^^^^^^^^^^^^^^^^^^^

For practical use cases, you want to optimize your own functions instead of running benchmark functions. Let's see how
we implement benchmark functions. As an example, :class:`baxus.benchmarks.real_world_benchmarks.MoptaSoftConstraints` implements
:class:`baxus.benchmarks.benchmark_function.SyntheticBenchmark`, which means in particular that it has its
own ``__call__`` function.

Let's look at the ``__call__`` function
of :class:`baxus.benchmarks.real_world_benchmarks.MoptaSoftConstraints` :

.. code-block:: python

   def __call__(self, x):
       super(MoptaSoftConstraints, self).__call__(x)
       x = np.array(x)
       if x.ndim == 0:
           x = np.expand_dims(x, 0)
       if x.ndim == 1:
           x = np.expand_dims(x, 0)
       assert x.ndim == 2

       vals = np.array([self._call(y) for y in x]).squeeze()
       return vals

which consists of some checks that ensure that we use the internal ``self._call`` function correctly.

If you want to use BAxUS with a custom function, you can just use this implementation and replace
``self._call`` in the line
``vals = np.array([self._call(y) for y in x]).squeeze()``
with a call to your own function expecting a 1D numpy array.

How do I register my new function?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this we need to look at the :class:`baxus.util.parsing.parse` function.
The first thing to do is to append your benchmark to the list of existing benchmarks,
currently consisting of

.. code-block:: python

   required_named.add_argument(
           "-f",
           "--functions",
           nargs="+",
           choices=[
               "hartmann6",
               "branin2",
               ...,
               "MY_NEW_NAME" # <---------------- ADD THIS LINE 
           ],
           required=True,
       )

Next, we have to register the new name in the :class:`baxus.util.parsing.fun_mapper>` function:

.. code-block:: python

   def fun_mapper():
       return {
           **{
               "hartmann6": Hartmann6,
               "branin2": Branin2,
               "rosenbrock2": functools.partial(RosenbrockEffectiveDim, effective_dim=2),
               ...,
               "MY_NEW_NAME": MyBenchmarkImplementation # <--------- ADD THIS LINE
           },
           **_fun_mapper,
       }

and that's it. 
