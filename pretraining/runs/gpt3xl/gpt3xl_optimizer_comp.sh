
# oai baseline
cd ~/picodo-bs
python main.py \
  +model=gpt3xl \
  +dataset=fw_gpt2 \
  opt.optimizer='adamw' \
  opt.batch_size=512 \
  opt.max_microbatch_size=16 \
  model.remat=True \
  opt.peak_lr=0.0002 \
  opt.b1=0.9 \
  opt.b2=0.95 \
  opt.weight_decay=0.1 \
  run_name='gpt3xl_oai_2'


# bs1 adam, fixed b2
cd ~/picodo-bs
python main.py \
  +model=gpt3xl \
  +dataset=fw_gpt2 \
  opt.optimizer='adamw' \
  opt.batch_size=1 \
  num_tp_devices=4 \
  opt.peak_lr=0.000067 \
  opt.b1=0.9 \
  opt.b2=0.95 \
  run_name='gpt3xl_adam_b2_2'

# bs1 adam, fixed t2
cd ~/picodo-bs
python main.py \
  +model=gpt3xl \
  +dataset=fw_gpt2 \
  opt.optimizer='adamw' \
  opt.batch_size=1 \
  num_tp_devices=4 \
  opt.peak_lr=0.000067 \
  opt.b1=0.9 \
  opt.b2=0.9999 \
  run_name='gpt3xl_adam_t2_2'

# bs1 sgd
cd ~/picodo-bs
python main.py \
  +model=gpt3xl \
  +dataset=fw_gpt2 \
  opt.optimizer='sgd' \
  opt.batch_size=1 \
  num_tp_devices=4 \
  opt.peak_lr=0.15 \
  run_name='gpt3xl_sgd_2'

# bs1 adafactor
cd ~/picodo-bs
python main.py \
  +model=gpt3xl \
  +dataset=fw_gpt2 \
  opt.optimizer='adafactor' \
  opt.batch_size=1 \
  num_tp_devices=4 \
  opt.peak_lr=0.0032 \
  opt.b2=0.9999 \
  run_name='gpt3xl_adafactor_2_0032'
