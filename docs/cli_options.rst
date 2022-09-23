
Command line options
--------------------

.. list-table::
   :widths: 15 10 25 10 40
   :class: longtable
   :header-rows: 1

   * - **Name**
     - **Shortcut**
     - **Full argument**
     - **Default**
     - **Description**
   * - Algorithm
     - -a
     - --algorithms
     - baxus
     - The algorithm to run. Has to be from baxus, embedded_turbo_target_dim, embedded_turbo_effective_dim, embedded_turbo_2_effective_dim, random_search
   * - Function
     - -f
     - --functions
     - None
     - One ore several test functions. Has to be from hartmann6, branin2, rosenbrock5, rosenbrock10, ackley, rosenbrock, levy, dixonprice, griewank, michalewicz, rastrigin, svm, lasso-high, lasso-dna, lasso-high, lasso-medium, lasso-leukemia, lasso-rcv1, lasso-breastcancer, lasso-diabetes, lasso-simple, lasso-hard, mopta08, hartmann6in1000_rotated, shiftedackley10.
   * - Input dimensionality
     - -id
     - --input-dim
     - 100
     - Input dimensionality of the function. This is overriden when the function has a fixed dimensionality.
   * - Target dimensionality
     - -td
     - --target-dim
     - 10
     - (Initial) target dimensionality of the function. Whether initial or not depends on the algorithm. Initial for ``BAxUS`` as it adapts the target dimensionality.
   * - Acquisition function
     - None
     - --acquisition-function
     - ts
     - Either ``ts`` (Thompson sampling) or ``ei`` (Expected improvement)
   * - Embedding type
     - None
     - --embedding-type
     - baxus
     - Either ``baxus`` (for the BAxUS embedding) or ``hesbo`` (for the HeSBO embedding)
   * - Adjust initial target dimensionality
     - None
     - --adjust-initial-target-dimension
     - not set
     - Whether to adjust the initial target dimensionality as described in the BAxUS paper.
   * - Number of initial samples
     - -n
     - --n-init
     - None (set to target dimensionality + 1 if not set)
     - Number of DOE samples.
   * - Number of repetitions
     - -r
     - --num-repetitions
     - 1
     - Number of repetitions of the run.
   * - Number of evaluations
     - -m
     - --max-evals
     - 300
     - Number of evaluations. Cma-ES might use a few more.
   * - Initial baselength
     - -l
     - --initial-baselength
     - 0.8
     - The initial base length of the trust region (default value is as in the TuRBO paper).
   * - Minimum baselength
     - -lmin
     - --min-baselength
     - 0.5^7
     - The minimum base length a trust region is allowed to obtain (default value is as in the TuRBO paper).
   * - Maximum baselength
     - -l_max
     - --max-baselength
     - 1.6
     - The maximum base length a trust region is allowed to obtain (default value is as in the TuRBO paper).
   * - Noise standard deviation
     - None
     - --noise-std
     - 0
     - The standard deviation of the noise. Whether this is used or not depends on the benchmark. It is generally only recognized for synthetic benchmarks like ``Branin2`` but also for the synthetic ``Lasso`` versions.
   * - Results directory
     - None
     - --results-dir
     - results
     - The directory to which the results are written. Relative to the path from which the run was started.
   * - Run description
     - None
     - --run-description
     - None
     - Short description that will be added to the run directory
   * - MLE multistart samples
     - None
     - --multistart-samples
     - 100
     - Number of multistart samples for the MLE GD optimization. Samples will be drawn from latin hypercube
   * - Multistarts after sampling
     - None
     - --multistart-after-sample
     - 10
     - Only recognized for '--mle-optimization sample-and-choose-best'. Number of multi-start gradient descent optimization out of the ``--multistart-samples`` best ones.
   * - MLE optimization method
     - None
     - --mle-optimization
     - sample-and-choose-best
     - Either ``multistart-gd`` or ``sample-and-choose-best``.
   * - Number of MLE gradient updates
     - None
     - --mle-training-steps
     - 50
     - Number of GD steps in MLE maximization.
   * - Budget until input dimensionality
     - None
     - --budget-until-input-dim
     - 0
     - The budget after which BAxUS will roughly reach the input dimensionality (see paper for details). If ``0``\ : this setting is ignored
   * - Verbose mode
     - -v
     - --verbose
     - not set
     - Whether to print verbose messages