#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

import sys, getopt

print('Hello its me torch ', torch.__version__)

