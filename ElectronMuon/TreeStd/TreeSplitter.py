import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

SRC = "../ONNX/RFMuonElectron.onnx"
DST = "../ONNX/RFMuonElectron_trees.onnx"

model = onnx.load(SRC)
node = next(n for n in model.graph.node if n.op_type == "TreeEnsembleClassifier")
attrs = {a.name: a for a in node.attribute}

def get(name):
    if name in attrs:
        return helper.get_attribute_value(attrs[name])
    if name + "_as_tensor" in attrs:
        return numpy_helper.to_array(helper.get_attribute_value(attrs[name + "_as_tensor"]))
    raise KeyError(name)

class_nodeids = np.array(get("class_nodeids"))
class_treeids = np.array(get("class_treeids"))
class_ids     = np.array(get("class_ids"))
class_weights = np.array(get("class_weights"), dtype=np.float64)

tree_ids_unique = sorted(set(int(t) for t in get("nodes_treeids")))
tree_index = {t: i for i, t in enumerate(tree_ids_unique)}
n_trees = len(tree_ids_unique)

print("unikalne class_ids:", sorted(set(class_ids.tolist())))
print("post_transform:", attrs["post_transform"].s.decode() if "post_transform" in attrs else "NONE")
print("min/max waga:", class_weights.min(), class_weights.max(), " 1/n_trees =", 1.0 / n_trees)

# Jedna waga na liść. Zwykle w[liść] = P(klasa)/n_trees (skl2onnx, agregacja SUM).
ids_present = set(class_ids.tolist())
if ids_present == {1}:
    flip = False          # wagi dotyczą klasy 1 (muon)
elif ids_present == {0}:
    flip = True           # wagi dotyczą klasy 0 -> P(mu) = 1 - P(e)
else:
    flip = False          # obie klasy zapisane -> bierzemy tylko klasę 1
    mask = class_ids == 1
    class_nodeids, class_treeids, class_weights = (
        class_nodeids[mask], class_treeids[mask], class_weights[mask])

# skala: jeśli wagi są już podzielone przez N drzew (max <= 1/N), mnożymy przez N
scale = n_trees if class_weights.max() <= 1.0 / n_trees + 1e-6 else 1.0
w = class_weights * scale
if flip:
    w = 1.0 - w
print(f"scale = {scale}, flip = {flip}, zakres wag po skalowaniu: {w.min():.4f} .. {w.max():.4f}")

t_ids = np.array([tree_index[int(t)] for t in class_treeids])

reg_attrs = {a.name: helper.get_attribute_value(a)
             for a in node.attribute if a.name.startswith("nodes_")}

reg = helper.make_node(
    "TreeEnsembleRegressor", [node.input[0]], ["tree_probs"],
    domain="ai.onnx.ml", name="per_tree",
    n_targets=n_trees,
    target_ids=t_ids.tolist(),
    target_nodeids=class_nodeids.tolist(),
    target_treeids=class_treeids.tolist(),
    target_weights=w.astype(np.float32).tolist(),
    aggregate_function="SUM",
    post_transform="NONE",
    **reg_attrs,
)
inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, [None, 5])
out = helper.make_tensor_value_info("tree_probs", TensorProto.FLOAT, [None, n_trees])
graph = helper.make_graph([reg], "rf_per_tree", [inp], [out])
m = helper.make_model(graph, opset_imports=model.opset_import)
m.ir_version = model.ir_version
onnx.checker.check_model(m)
onnx.save(m, DST)
print("zapisano", DST)