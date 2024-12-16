prepare-cifar10:
	./dataset/download_cifar.sh

run-app:
	poetry run streamlit run python/app.py
