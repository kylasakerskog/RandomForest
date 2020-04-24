## モチベーション
- Random Forestの原理を実装を通して理解する
  - [このサイト](https://qiita.com/deaikei/items/52d84ccfedbfc3b222cb)を参考に実装
  - 消えた時用の[魚拓](http://web.archive.org/web/20170905205619/http://qiita.com:80/deaikei/items/52d84ccfedbfc3b222cb)

## What is RandomForest
- 決定木(Decision Tree)をたくさん作って予測を行う手法
- 決定木を用いたバギング
  - バギング : 複数のモデルを独自に学習させて，それぞれのモデルの出力を多数決して出力を決める．  

## What is Decision Tree
- 条件分岐の繰り返しによって分類問題を解く教師あり学習のモデル
  - 各ノードでデータを分割
- 利点 : 意味解釈性が高く，前処理が要らない，さらに学習が高速
- 欠点 : 柔軟さがない，過学習しやすい
- 分岐がきれいか否かを示す不純度を目的関数に
  - 不純度の減り方が最大となる分割手法で分割すべし

## USAGE:

```
$ python3 random_forest.py
```
