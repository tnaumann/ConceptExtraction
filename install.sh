#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p "$DIR/lib"
cd "$DIR/lib"

# Get libsvm
wget "http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip" -O "libsvm-3.14.zip"
unzip "libsvm-3.14.zip"

ln -s "libsvm-3.14" "libsvm"
rm "libsvm-3.14.zip"

cd "libsvm"
make

# Get liblinear
wget "http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/liblinear.cgi?+http://www.csie.ntu.edu.tw/~cjlin/liblinear+zip" -O "liblinear-1.92.zip"
unzip "liblinear-1.92.zip"

ln -s "liblinear-1.92" "liblinear"
rm "liblinear-1.92.zip"

cd "liblinear"
make