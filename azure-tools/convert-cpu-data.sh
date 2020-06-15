#!/bin/sh
#
# Converts Azure's CPU data to per user data.
#
# Author: Liran Funaro <liran.funaro@gmail.com>
#
# Copyright (C) 2006-2018 Liran Funaro
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Stop on error
set -e

# Count lines
# zcat vm_cpu_readings-file-1-of-125.csv.gz | wc -l

# Count output files
# ls -l azure_cpu_data/ | wc -l

# VM ID characters
# '+/0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

dataset_path=$1
vm_id_set_fname=$2
cpu_data_dname=$3

echo "Compiling..."
g++ convert-cpu-data.cpp -o bin/convert-cpu-data

echo "Remove old data..."
rm -fr ${cpu_data_dname}

echo "Create folder..."
mkdir -p ${cpu_data_dname}

for i in `seq 1 125`; do
	echo "File $i: vm_cpu_readings-file-$i-of-125.csv.gz"
	zcat ${dataset_path}/vm_cpu_readings-file-${i}-of-125.csv.gz  | bin/convert-cpu-data \
	        ${vm_id_set_fname} ${cpu_data_dname}
done
