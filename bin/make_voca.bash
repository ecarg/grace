#!/usr/bin/env bash
basedir=$(dirname "$0")
input_file=${basedir}/../data/v3.train
output_dir=${basedir}/../rsc

# 음절단위 입력 사전 
${basedir}/make_voca.py --input=${input_file} \
    --unit=syllable --type=in --output=${output_dir}/voca.syl.in

# 출력 사전
${basedir}/make_voca.py --input=${input_file} \
    --unit=syllable --type=out --output=${output_dir}/voca.out

# 자소단위 입력 사전
${basedir}/make_voca.py --input=${input_file} \
    --unit=phonemes --type=in  --output=${output_dir}/voca.pho.in
