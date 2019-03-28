
python3 main.py --cuda --model GRU   --nlayers 1 --epochs 10 --emsize 200 2>&1 | tee experiments.report # 430  57.85  4.95  4.88
python3 main.py --cuda --model GRU   --nlayers 1 --epochs 10 --emsize 400 2>&1 | tee experiments.report # 471  50.22  4.89  4.82
python3 main.py --cuda --model GRU   --nlayers 1 --epochs 10 --emsize 800 2>&1 | tee experiments.report # 510  44.74  4.91  4.85
python3 main.py --cuda --model GRU   --nlayers 1 --epochs 20 --emsize 800 2>&1 | tee experiments.report # 1014 42.18  4.88  4.82
python3 main.py --cuda --model GRU   --nlayers 2 --epochs 20 --emsize 800 2>&1 | tee experiments.report # 1252 209.7  224.86  191.50
python3 main.py --cuda --model LSTM  --nlayers 1 --epochs 10 --emsize 200 2>&1 | tee experiments.report # 433  58.82  128.36  121.06
python3 main.py --cuda --model LSTM  --nlayers 1 --epochs 10 --emsize 400 2>&1 | tee experiments.report # 480  50.96  133.88  124.72
python3 main.py --cuda --model LSTM  --nlayers 1 --epochs 10 --emsize 800 2>&1 | tee experiments.report # 518  47.08  134.31  125,84
python3 main.py --cuda --model LSTM  --nlayers 1 --epochs 20 --emsize 800 2>&1 | tee experiments.report # 1046 39.61  126.91  119.7
python3 main.py --cuda --model LSTM  --nlayers 2 --epochs 20 --emsize 800 2>&1 | tee experiments.report # 1297 52.51  116.33  110.21
