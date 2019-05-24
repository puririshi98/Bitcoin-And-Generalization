to get data
./data_download.sh # will download it to /tmp/
\n
python3 data_generator.py big_data /tmp/coinbaseUSD.csv 
\n

to train the resnet simply run 
\n
python3 train_stock.py
\n

now to embed the data in the 512 features of resnet run
\n
python3 embed_512.py 
\n
to evaluate
\n

next we to train an ensemble of 15 512-18-2(winning size) networks
\n
python3 train_embed_ensemble.py
\n

to evaluate on unseen data run 
\n
python3 data_generator.py final_unseen_test /tmp/coinbaseUSD.csv 
\n
python3 embed_unseen.py
\n
python3 ensemble_eval.py







