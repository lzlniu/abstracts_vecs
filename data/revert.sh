#!/bin/bash
awk -F '\t' '{print $1"\t"1/$2}' $1
