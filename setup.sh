#!/bin/bash

[ -d venv ] || python3.11 -m venv venv
pip install -r requirements.txt
