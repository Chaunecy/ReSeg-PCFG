This is the source code of ReSeg-PCFG.

## Requirements + Installation
- Python3 is the only hard requirement for these tools
- It is **highly recommended** that you install the chardet python3 library for training. While not required, it performs character encoding autodetection of the training passwords. To install it:
 - Download the source from [https://pypi.python.org/pypi/chardet](http://https://pypi.python.org/pypi/chardet "https://pypi.python.org/pypi/chardet")
 - Or install it using `pip3 install chardet`

## Training

Using `trainer.py` to train a model.

```
python trainer.py -t <password file> -r <rule folder pathname> -n 2 -c 1
```

Note that we use `-n 2 -c 1` to avoid using OMEN.
## Monte Carlo 

Using `scorer.py` to calculate the guess numbers of passwords.

```
python scorer.py -r <rule folder pathname> -t <password file> -s <results filename> -n <number of password samples>
```
