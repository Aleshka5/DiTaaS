sudo apt update
sudo apt install python3.13
sudo apt install python3.13-pip
sudo apt install python3.13-venv
python3.13 -m venv .venv
source .venv/bin/activate
pip install uv
uv run --env-file .env -m cli.load_dataset --output-dir ./dataset
