#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p "$DIR/lib"
cd "$DIR/lib"

wget "http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip" -O "libsvm-3.14.zip"
unzip "libsvm-3.14.zip"

ln -s "libsvm-3.14" "libsvm"

rm "libsvm-3.14.zip"