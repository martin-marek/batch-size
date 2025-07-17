
# bs512, adamw
python main.py \
  +model=gpt2s \
  +dataset=fw_gpt2 \
  opt.optimizer='adamw' \
  opt.batch_size=512 \
  opt.max_microbatch_size=16 \
  model.remat=True \
  opt.peak_lr=0.0048 \
  opt.b1=0.9 \
  opt.b2=0.95 \
  opt.weight_decay=0.1

# bs1 adam, fixed b2
python main.py \
  +model=gpt2s \
  +dataset=fw_gpt2 \
  opt.optimizer='adamw' \
  opt.batch_size=1 \
  opt.peak_lr=0.00015 \
  opt.b1=0.9 \
  opt.b2=0.95

# bs1 adam, fixed t2
python main.py \
  +model=gpt2s \
  +dataset=fw_gpt2 \
  opt.optimizer='adamw' \
  opt.batch_size=1 \
  opt.peak_lr=0.0024 \
  opt.b1=0.9 \
  opt.b2=0.9999

# bs1 sgd
python main.py \
  +model=gpt2s \
  +dataset=fw_gpt2 \
  opt.optimizer='sgd' \
  opt.batch_size=1 \
  opt.peak_lr=0.2

# bs1 adafactor
python main.py \
  +model=gpt2s \
  +dataset=fw_gpt2 \
  opt.optimizer='adafactor' \
  opt.batch_size=1 \
  opt.peak_lr=0.005 \
  opt.b2=0.9999
