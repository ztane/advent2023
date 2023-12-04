#!/usr/bin/env python3
import sys

if len(sys.argv) != 2:
    print('Usage: runner.py daynumber')
    sys.exit(2)

day = int(sys.argv[1])
day_module = 'day{:02}'.format(day)
module = getattr(__import__('days.' + day_module), day_module)
