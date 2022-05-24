#!/usr/bin python3
# -*- coding: utf-8 -*-
import subprocess, shlex, os

if __name__ == '__main__':
    print("*"*5)
    print("hello world!")
    os.system("./temp.sh")
    # subprocess.Popen(shlex.split("./temp.sh"))
    print("done with temp.sh")