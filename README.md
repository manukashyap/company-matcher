# Company Matcher

## What it does
Company matcher can be used to find closest match of various companies from a preprocessed list of companies. 
It does exact matching of names alongside fuzzy matching if no exact matches are found. 
Companies are ranked in the order of highest simiilarity score. Score calculation is done on the basis of matching parameters (exact and fuzzy) and priorities assingned to them.

## How to run
- Install and setup conda on your local machine [Link Here](https://docs.anaconda.com/miniconda/miniconda-install/)
- Create a new environment using ``` conda create -n comp_match_env  python=3.9.13 anaconda ```
- Activate environment using ``` conda activate comp_match_env ```
- Install the requirements
- Run main.py

