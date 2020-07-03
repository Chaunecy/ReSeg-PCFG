#!/usr/bin/env bash
set -e
set -u
set -x
/home/cw/Codes/Python/pcfg_cracker/venv/bin/python \
    /home/cw/Codes/Python/pcfg_cracker/trainer.py \
    -t /home/cw/Codes/Python/PwdTools/corpora/src/csdn-src.txt \
    -r ./Rules/Origin/csdn \
    --save-seg /home/cw/Documents/Expirements/SegLab/Segments/PCFGv41/csdn.txt \
    -n 2 -c 1

/home/cw/Codes/Python/pcfg_cracker/venv/bin/python \
    /home/cw/Codes/Python/pcfg_cracker/trainer.py \
    -t /home/cw/Codes/Python/PwdTools/corpora/src/rockyou-src.txt \
    -r ./Rules/Origin/rockyou \
    --save-seg /home/cw/Documents/Expirements/SegLab/Segments/PCFGv41/rockyou.txt \
    -n 2 -c 1

/home/cw/Codes/Python/pcfg_cracker/venv/bin/python \
    /home/cw/Codes/Python/pcfg_cracker/trainer.py \
    -t /home/cw/Codes/Python/PwdTools/corpora/src/webhost-src.txt \
    -r ./Rules/Origin/webhost \
    --save-seg /home/cw/Documents/Expirements/SegLab/Segments/PCFGv41/webhost.txt \
    -n 2 -c 1


/home/cw/Codes/Python/pcfg_cracker/venv/bin/python \
    /home/cw/Codes/Python/pcfg_cracker/trainer.py \
    -t /home/cw/Codes/Python/PwdTools/corpora/src/dodonew-src.txt \
    -r ./Rules/Origin/dodonew \
    --save-seg /home/cw/Documents/Expirements/SegLab/Segments/PCFGv41/dodonew.txt \
    -n 2 -c 1

/home/cw/Codes/Python/pcfg_cracker/venv/bin/python \
    /home/cw/Codes/Python/pcfg_cracker/trainer.py \
    -t /home/cw/Codes/Python/PwdTools/corpora/src/xato-src.txt \
    -r ./Rules/Origin/xato \
    --save-seg /home/cw/Documents/Expirements/SegLab/Segments/PCFGv41/xato.txt \
    -n 2 -c 1

