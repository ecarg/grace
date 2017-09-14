#!/usr/bin/env bash
basedir=$(dirname "$0")

# 음절단위 입력 사전 
${basedir}/make_voca.py --input=${basedir}/../data/corpus.train\
    --unit=syllable --type=in --output=${basedir}/../rsc/voca.syl.in

# 음절단위 출력 사전
${basedir}/make_voca.py --input=${basedir}/../data/corpus.train\
    --unit=syllable --type=out --output=${basedir}/../rsc/voca.syl.out

# 자소단위 입력 사전
${basedir}/make_voca.py --input=${basedir}/../data/corpus.train\
    --unit=phonemes --type=in  --output=${basedir}/../rsc/voca.pho.in

# 자소단위 출력 사전(음절단위와 동일하다)
${basedir}/make_voca.py --input=${basedir}/../data/corpus.train\
    --unit=phonemes --type=out  --output=${basedir}/../rsc/voca.pho.out



