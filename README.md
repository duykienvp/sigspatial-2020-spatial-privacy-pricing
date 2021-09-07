# Code for SIGSPATIAL 2020 paper on Spatial Privacy Pricing
Paper: Spatial Privacy Pricing: The Interplay between Privacy, Utility and Price in Geo-Marketplaces (https://dl.acm.org/doi/10.1145/3397536.3422213)

## Getting Started
1. Dependency
  - Python 3.7.5

2.	Installation process
  - Upgrade pip: `pip install --upgrade pip`
  - Upgrade setuptools: `pip install --upgrade setuptools`
  - Install dependency: `pip install -r requirements.txt`
  - Create logs directory: e.g. `mkdir logs`
  - Create output directory: e.g. `mkdir -p output/current`
  - Make sure `data` link point to the correct data directory: e.g. `ln -s /path/to/data/ data`

## Usage
- Use the `run.py` to run the program: `python run.py --task <task_name> --approach <approach_name>`
- Other parameters are provided in `config.yml` file.


### Example running steps:
- Collect check-in data for an area: 
  - This process loads all data from a file, collect data points inside an area, and save them to a file.
  This file can be used later so that we do not need to load the big data file.
  - Set up configuration with data_file as input file, output_file as output file, dataset_type and area_code
  - Execute:
  ```
    python run.py --task collect
  ```
  - We have a gowalla_LA.csv file as an example of Gowalla data from SNAP dataset.
  
- Decision Theoretic Problem:
  To run the program for decision theoretic problem (with payoff matrix), prepare configuration, 
  then use the following examples:
  - Run Ground Truth algorithm:
    ```
      python run.py --task benchmark --approach gt
    ```
  - Run Uniform Prior algorithm:
    ```
    python run.py --task benchmark --approach up
    ```
  - Run Fixed Maximum Cost algorithm:
    ```
    python run.py --task benchmark --approach fmc
    ```
  - Run Probing algorithm:
    ```
    python run.py --task benchmark --approach probing
    ```
    For probing approach, `algorithm` value in `config.yml` file specifies the probing algorithm. 
    Use `POI` for POI algorithm, or `SIP` for SIP and SIP-T algorithm. SIP is run when `should_check_inout` is `False`. 
    SIP-T is run when `should_check_inout` is `True`.  
  
# Contribute
- Kien Nguyen (duykienvp@gmail.com)
