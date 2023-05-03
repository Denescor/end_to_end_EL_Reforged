#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 09:54:11 2022

@author: carpentier
"""

import model.train as train

train_args = train._parse_args()
train.log_args(train_args, train_args.output_folder+"train_args.txt")
print("train args generated")