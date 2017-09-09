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

# make_training.py
* 학습데이터로 사용하기 위해서 raw 코퍼스를 고정길이 컨텍스트를 갖는 형태로 변환합니다.
* 사용법
```
$ cat sample.txt
<김기표:PS>가 우수투수상, <이지영:PS>(이상 <경성대:OG>)이

$ ./make_training.py -m ./ --input=./sample.txt --context-size=10
B-PS    <p> <p> <p> <p> <p> <p> <p> <p> <p> <w> 김 기 표 가 </w> 우 수 투 수 상 ,
I-PS    <p> <p> <p> <p> <p> <p> <p> <p> <w> 김 기 표 가 </w> 우 수 투 수 상 , 이
I-PS    <p> <p> <p> <p> <p> <p> <p> <w> 김 기 표 가 </w> 우 수 투 수 상 , 이 지
O    <p> <p> <p> <p> <p> <p> <w> 김 기 표 가 </w> 우 수 투 수 상 , 이 지 영
O    <p> <p> <p> <p> <p> 김 기 표 가 <w> 우 수 투 수 상 , </w> 이 지 영 (
O    <p> <p> <p> <p> 김 기 표 가 <w> 우 수 투 수 상 , </w> 이 지 영 ( 이
O    <p> <p> <p> 김 기 표 가 <w> 우 수 투 수 상 , </w> 이 지 영 ( 이 상
O    <p> <p> 김 기 표 가 <w> 우 수 투 수 상 , </w> 이 지 영 ( 이 상 경
O    <p> 김 기 표 가 <w> 우 수 투 수 상 , </w> 이 지 영 ( 이 상 경 성
O    김 기 표 가 <w> 우 수 투 수 상 , </w> 이 지 영 ( 이 상 경 성 대
B-PS    기 표 가 우 수 투 수 상 , <w> 이 지 영 ( 이 상 </w> 경 성 대 )
I-PS    표 가 우 수 투 수 상 , <w> 이 지 영 ( 이 상 </w> 경 성 대 ) 이
I-PS    가 우 수 투 수 상 , <w> 이 지 영 ( 이 상 </w> 경 성 대 ) 이 :
O    우 수 투 수 상 , <w> 이 지 영 ( 이 상 </w> 경 성 대 ) 이 : w
O    수 투 수 상 , <w> 이 지 영 ( 이 상 </w> 경 성 대 ) 이 : w q
O    투 수 상 , <w> 이 지 영 ( 이 상 </w> 경 성 대 ) 이 : w q </p>
B-OG    수 상 , 이 지 영 ( 이 상 <w> 경 성 대 ) 이 : w q </w> </p> </p>
I-OG    상 , 이 지 영 ( 이 상 <w> 경 성 대 ) 이 : w q </w> </p> </p> </p>
I-OG    , 이 지 영 ( 이 상 <w> 경 성 대 ) 이 : w q </w> </p> </p> </p> </p>
O    이 지 영 ( 이 상 <w> 경 성 대 ) 이 : w q </w> </p> </p> </p> </p> </p>
O    지 영 ( 이 상 <w> 경 성 대 ) 이 : w q </w> </p> </p> </p> </p> </p> </p>
O    영 ( 이 상 <w> 경 성 대 ) 이 : w q </w> </p> </p> </p> </p> </p> </p> </p>
O    ( 이 상 <w> 경 성 대 ) 이 : w q </w> </p> </p> </p> </p> </p> </p> </p> </p>
O    이 상 <w> 경 성 대 ) 이 : w q </w> </p> </p> </p> </p> </p> </p> </p> </p> </p>
```

# make_voca.bash
* 위 make_training.py에 의해 생성한 학습 코퍼스로부터 input/output vocabulary를 생성합니다.
* 사용법
```bash
./make_voca.bash train.chr rsc
```

# train.py
* 학습을 수행합니다.
* 사용법
```
usage: train.py [-h] -r DIR -i DIR -p NAME -m NAME -o FILE [--log FILE]
                [--window INT] [--embed-dim INT] [--gpu-num INT]
                [--batch-size INT] [--epoch-num INT] [--debug]

train model from data

optional arguments:
  -h, --help            show this help message and exit
  -r DIR, --rsc-dir DIR
                        resource directory
  -i DIR, --in-dir DIR  input directory
  -p NAME, --in-pfx NAME
                        input data prefix
  -m NAME, --model NAME
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
bin/train.py -r rsc -i data -p chr -m fnn -o fnn.pkl --log fnn.log
```
