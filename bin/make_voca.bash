#!/usr/bin/env bash


#############
# functions #
#############
function make_voca_out() {
    local in_file=$1
    local out_file=$2
    grep -v "^$" "${in_file}" | cut -f1 | LANG=C sort --buffer-size=1g --unique > "${out_file}"
}


function make_voca_in() {
    local in_file=$1
    local out_file=$2
    cut -f2 "${in_file}" | tr ' ' '\n' | grep -v "^$" | LANG=C sort --buffer-size=1g --unique > "${out_file}"
}


########
# main #
########
if [ $# -lt 2 ]; then
    echo "usage: $(basename "$0") [input file] [output dir]"
    exit 1
fi
in_file=$1
out_dir=$2


mkdir -p "${out_dir}"
make_voca_out "${in_file}" "${out_dir}"/voca.out
make_voca_in "${in_file}" "${out_dir}"/voca.in
