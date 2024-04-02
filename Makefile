#!make

include .env
export

download: venv
	$(VENV)/python src/data/download_dataset.py

prepare: download
	$(VENV)/python src/data/process_raw_dataset.py

train: prepare
	$(VENV)/python src/models/train_model.py

include Makefile.venv
Makefile.venv:
	curl \
		-o Makefile.fetched \
		-L "https://github.com/sio/Makefile.venv/raw/v2023.04.17/Makefile.venv"
	echo "fb48375ed1fd19e41e0cdcf51a4a0c6d1010dfe03b672ffc4c26a91878544f82 *Makefile.fetched" \
		| sha256sum --check - \
		&& mv Makefile.fetched Makefile.venv
