## Getting started

The main file is `benchmark_runner.py` in the project root.
It can be configured with command line arguments (see [Command Line Options](cli_options.html))

For example, to run `BAxUS` for 1,000 function evaluations on a Branin2 function with input dimensionality 100 for one 
repetition run

```
python3 benchmark_runner.py -id 100 -td 1 -n 10 -r 1 -m 1000 -f branin2 -a baxus --adjust-initial-target-dimension
```

Note that we need to pass an initial target dimensionality with `-td 1` even though this is adjusted later by passing 
the option `--adjust-initial-target-dimension`-