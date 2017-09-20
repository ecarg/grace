#!/usr/bin/env bash
basedir=$(dirname "$0")
input_file=${basedir}/../data/v2.train
output_dir=${basedir}/../rsc

# 음절단위 입력 사전 
${basedir}/make_voca.py --input=${input_file} \
    --unit=syllable --type=in --output=${output_dir}/voca.syl.in

# 음절단위 출력 사전
${basedir}/make_voca.py --input=${input_file} \
    --unit=syllable --type=out --output=${output_dir}/voca.syl.out

# 자소단위 입력 사전
${basedir}/make_voca.py --input=${input_file} \
    --unit=phonemes --type=in  --output=${output_dir}/voca.pho.in

# 자소단위 출력 사전(음절단위와 동일하다)
${basedir}/make_voca.py --input=${input_file} \
    --unit=phonemes --type=out  --output=${output_dir}/voca.pho.out
