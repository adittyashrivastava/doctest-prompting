#!/bin/bash
pip3 install --upgrade yapf
python3 -m yapf -ir -vv --style ./.style.yapf verl tests examples recipe scripts data
