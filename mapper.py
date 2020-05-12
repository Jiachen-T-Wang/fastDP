#!/usr/bin/env python3
import sys


for line in sys.stdin:
    segments = line.strip().split(",")
    if segments[0] == '"sex"':
        continue
    if 'NA' not in segments:
        a = segments[0:5]
        b = segments[6:10]
        a.extend(b)
        print(str(a) + "\t" + segments[5])

