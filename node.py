# 2分決定木のノード，class_count は必ず持つ．Leafでなければそれ以外も持つ
# このノードを連結していくことで2分決定木が作れる
class _Node:
    def __init__(self, class_count):
        self.class_count = class_count # このノードに属す各クラスのサンプル数

        self.feature_id = None # 分割規則に用いたのは「入力Xの何次元目か」のインデックス (int)
        self.threshold = None # 分割規則に用いたしきい値
        self.left = None # 左の子ノード (x < threshold)
        self.right = None # 右の子ノード(x >= threshold)

    @property #(@feature_id.getter)
    def is_leaf(self):
        return self.feature_id is None
        
    def __repr__(self):
        if self.is_leaf:
            return "<Node leaf is class_count: {}>".format(self.class_count)
        else:
            return "<Node is class_count: {}, feature>".format(
                self.class_count, self.feature_id, self.threshold
            )
        
