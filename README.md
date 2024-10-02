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

## How it works
- We begin by preprocesssing the input data, i.e. removing prefixes and suffixes from website url, converting company
country names to lower cases and removing stray spaces
- We then try to generate sensible embedding which effectively represent the information in each row of our data
- Here, we're using a combination of semantics based model and knowledge model
- We generate embeddings using both of these and then take a composition of the two (increasing the dimensions but storing better info)
- Now, we insert the embeddings from the larger dataset into our vector DB : ChromaDB in this instance
- Once, done we iterate over all elements from our smaller dataset and for each row, generate the embeddings in a similar fashion
- For each row from smaller dataset, we find the top_k nearest matches from the larger dataset

## Improvements
- Train a custom entity linker using a knowledge base and an annotation tool that can be used by humans to correctly label stuff
- Accumulate documents (data) related to present companies to try to enhance their vector repn
- Use better(fine-tuned on company data) or bigger models to capture more context


