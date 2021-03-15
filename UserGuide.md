ReSeg-PCFG
=====

We run the code using a machine with Ubuntu 20.04 LTS Desktop, Intel i7-8700 3.20GHz * 12, 1.0 TB disk, Python 3.8.5,
and gcc 9.3.0.

Trainer
----
Trainer (trainer.py) requires a training dataset and a directory to save the trained model.

In our experiments, we set `-n 2 -c 1` to disable OMEN feature.

The training phase takes from an hour (Dodonew) to five hours (Youku).

```shell
python3 trainer.py -t {filename of training dataset} -r {directory of trained model} -n 2 -c 1
```

Simulator
----
Simulator (scorer.py) requires a testing dataset, a trained model, sample size and directory of results.

In our experiments, the sample size is 1,000,000.

The simulating phase takes about an hour for each dataset.

```shell
python3 scorer.py -t {filename of testing dataset} -r {directory of trained model} --n-sample 1000000 -s {filename of saved results}
```

Enumerator
----
Enumerator (reseg_pcfg_guesser) requires a trained model, a maximal guess number, and a file to save enumerated guesses.

The enumerating phase enumerate about 4,000,000 guesses per second.

```shell
reseg_pcfg_guesser -r {trained model} -n 10000000000 -f {filename of enumerated guesses}
```