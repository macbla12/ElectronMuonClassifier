import onnx

model = onnx.load("../ONNX/RFMuonElectron.onnx")

for node in model.graph.node:
    if node.op_type == "TreeEnsembleClassifier":

        attrs = {
            attr.name: onnx.helper.get_attribute_value(attr)
            for attr in node.attribute
        }

        print("Number of trees:",
              len(set(attrs["nodes_treeids"])))

        print("Number of nodes:",
              len(attrs["nodes_treeids"]))

        print("Number of class weights:",
              len(attrs["class_weights"]))

        print("Output names:", list(node.output))