# corpus_converter.py
* 여러 포맷의 코퍼스를 엑소브레인 코퍼스 형태로 변환합니다.
* 사용법
```
usage: corpus_converter.py [-h] -f NAME [--input FILE] [--output FILE]
                           [--debug]

각종 포맷의 코퍼스를 엑소브레인 코퍼스 포맷으로 변환하는 스크립트

optional arguments:
  -h, --help            show this help message and exit
  -f NAME, --format NAME
                        input file format <json, exo, train>
  --input FILE          input file <default: stdin>
  --output FILE         output file <default: stdout>
  --debug               enable debug
```

# parser.py
* 다음과 같은 형식을 가지는 raw 코퍼스를 파싱합니다.
* 출력은 파싱 후, 원문을 그대로 출력합니다.
* 사용법
```
$ cat sample.txt
이날 결승타를 포함 4타수 2안타를 기록하는 등 이번 대회에서 타율 0.500(14타수 7안타)의 불방망이를 휘두른 <김강석:PS>은 타격상, <김기표:PS>가 우수투수상, <이지영:PS>(이상 <경성대:OG>)이 수훈상 기쁨을 각각 누렸고 8타점을 올린 <박민철:PS>(<한양대:OG>)은 타점상을 받았다.

$ ./parser.py --input=./sample.txt
이날 결승타를 포함 4타수 2안타를 기록하는 등 이번 대회에서 타율 0.500(14타수 7안타)의 불방망이를 휘두른 <김강석:PS>은 타격상, <김기표:PS>가 우수투수상, <이지영:PS>(이상 <경성대:OG>)이 수훈상 기쁨을 각각 누렸고 8타점을 올린 <박민철:     PS>(<한양대:OG>)은 타점상을 받았다.
```

# make_voca.bash
* 엑소브레인 코퍼스로부터  input/output vocabulary를 생성합니다.
* 사용법
```bash
./make_voca.bash
```

# train.py
* 학습을 수행합니다.
* 사용법
```
usage: train.py [-h] -r DIR -p NAME -m NAME -o FILE [--log FILE]
[--window INT] [--embed-dim INT] [--gpu-num INT]
[--batch-size INT] [--epoch-num INT] [--debug]

train model from data

optional arguments:
  -h, --help            show this help message and exit
  -r DIR, --rsc-dir DIR
                        resource directory
  -p NAME, --in-pfx NAME
                        input data prefix
  -m NAME, --model-name NAME
                        model name
  -o FILE, --output FILE
                        model output file
  --log FILE            loss and accuracy log file
  --window INT          left/right character window length <default: 10>
  --embed-dim INT       embedding dimension <default: 50>
  --gpu-num INT         GPU number to use <default: 0>
  --batch-size INT      batch size <default: 100>
  --epoch-num INT       epoch number <default: 100>
  --debug               enable debug
```


* 사용예
```bash
bin/train.py -r rsc -p data/corpus -m fnn -o fnn.out  --log fnn.log
```

# tag.py
* 입력된 텍스트에 개체명 태깅을 수행합니다.
* 사용법1) 개체명 태깅을 수행하고 싶은 경우
```
$ cat ./sample.txt
이날 에인절스 선발 존 래키를 상태로 최희섭이
$ ./tag.py -i ./sample -m model_path
이날에 <인절스:OG> 선발 <존 래키:PS>를 상대로 <최희섭:PS>이

```
* 사용법2 태깅된 코퍼스를 이용하여 정확률 / 재현률을 체크하고 싶은 경우
* 태깅된 원문도 같이 출력됩니다.
$ cat ./tagged.txt
이날에 <인절스:OG> 선발 <존 래키:PS>를 상대로 <최희섭:PS>은
$ ./tag.py -i ./tagged.txt -m model_path --eval
이날에 <인절스:OG> 선발 <존 래키:PS>를 상대로 <최희섭:PS>은
accuracy: 0.902757, f-score: 0.366470 (recall = 0.387546, precision = 0.347569
```

