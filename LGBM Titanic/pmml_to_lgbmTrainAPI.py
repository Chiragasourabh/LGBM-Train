from pprint import pprint
from PMML44 import *
import numpy as np
import pandas as pd
import lightgbm as lgb
import os

def reconstruct(pmml_file_name):
    

    def get_tree_string(segmentation=None, temp_file = None):
        segment_list = segmentation.get_Segment()
        for segment in segment_list:
            append_tree_data_to_file(segment=segment,file=temp_file)
            temp_file.write("\n\n")
        temp_file.write("end of trees")
        return temp_file

    def append_tree_data_to_file(segment=None,file=None):
        tree_id = str(int(segment.get_id())-1)
        file.write("Tree="+tree_id+"\n")
        get_string_tree_data(segment=segment,file=file)
        return None

    def get_string_tree_data(segment=None,file=None):
        tree_model = segment.get_TreeModel()
        root_node = tree_model.get_Node()

        split_feature=dict()
        split_gain=dict()
        threshold=dict()
        decision_type=dict()
        left_child=dict()
        right_child=dict()
        leaf_value=dict()
        leaf_count=dict()
        internal_value=dict()
        internal_count=dict()

        def get_child_node_data(parent_node = root_node):
            if parent_node is not None:
                gain , node_info = get_extension_and_node_info(Node = parent_node)
                node_id = parent_node.get_id()
                   
                if node_info:
                    split_gain[node_id] = gain
                    internal_count[node_id] = int(parent_node.get_recordCount())
                    internal_value[node_id] = parent_node.get_score()

                    #Not Sure About dictionary keys order ------- STARTS FROM HERE
                    SimplePredicate = parent_node.get_Node()[0].get_SimplePredicate()
                    if SimplePredicate is not None:
                        split_feature[node_id] = features.index(SimplePredicate.get_field())
                        decision_type[node_id] = 2 #SimplePredicate.get_operator() based on this
                        threshold[node_id] = SimplePredicate.get_value()
                    #ENDS HERE

                else:
                    leaf_count[node_id] = int(parent_node.get_recordCount())
                    leaf_value[node_id] = parent_node.get_score()

                child_nodes = parent_node.get_Node()
                for node in child_nodes:
                    get_child_node_data(node)
            
        def get_extension_and_node_info(Node = None):
            ExtensionList = Node.get_Extension()
            intermediate_node = False
            if ExtensionList:
                intermediate_node = True
                if ExtensionList[0].get_name() == 'gain':
                    return np.float64(ExtensionList[0].get_value()), intermediate_node
            return 0, intermediate_node

        get_child_node_data(parent_node = root_node)
        split_feature_list = [split_feature[i] for i in sorted (split_feature)]
        split_gain_list = [split_gain[i] for i in sorted (split_gain)]
        threshold_list = [threshold[i] for i in sorted (threshold)]
        decision_type_list = [decision_type[i] for i in sorted (decision_type)]
        leaf_value_list = [leaf_value[i] for i in sorted (leaf_value)]
        leaf_count_list = [leaf_count[i] for i in sorted (leaf_count)]
        internal_value_list = [internal_value[i] for i in sorted (internal_value)]
        internal_count_list = [internal_count[i] for i in sorted (internal_count)]
        shrinkage = tree_model.get_Extension()[0].get_value()

        file.write("num_leaves="+str(len(leaf_count))+"\n")
        file.write("num_cat="+"0"+"\n") #Not sure
        file.write("split_feature="+" ".join(map(str, split_feature_list))+"\n")
        file.write("split_gain="+" ".join(map(str, split_gain_list))+"\n")
        file.write("threshold="+" ".join(map(str, threshold_list))+"\n")
        file.write("decision_type="+" ".join(map(str, decision_type_list))+"\n")
        file.write("left_child="+"-1 -2"+"\n") #Hardcoded False values
        file.write("right_child="+"1 -3"+"\n") #Hardcoded False values
        # file.write("left_child="+"Null"+"\n")
        # file.write("right_child="+"Null"+"\n")
        file.write("leaf_value="+" ".join(map(str, leaf_value_list))+"\n")
        file.write("leaf_count="+" ".join(map(str, leaf_count_list))+"\n")
        file.write("internal_value="+" ".join(map(str, internal_value_list))+"\n")
        file.write("internal_count="+" ".join(map(str, internal_count_list))+"\n")
        file.write("shrinkage="+shrinkage+"\n")
        print(" ".join(map(str, split_gain_list)))
        
    nyoka_pmml = parse(pmml_file_name, silence=True)
    mining_model_obj = nyoka_pmml.MiningModel[0]
    mf = mining_model_obj.get_MiningSchema().get_MiningField()
    features = list()
    feature_infos = list()
    for field in mf:
        features.append(field.get_name())
        feature_infos.append("["+str(field.get_lowValue())+":"+str(field.get_highValue())+"]")
    segmentation_obj = mining_model_obj.Segmentation
    filename = "tempfile.txt"
    f = open(filename, "w+")
    f.write("tree\n"+
                    "version=v2\n"+
                    "num_class=1\n"+
                    "num_tree_per_iteration=1\n"+
                    "label_index=0\n"+
                    "max_feature_idx="+str(len(features)-1)+"\n"+
                    "objective=regression\n"+   #Have to change it logically
                    "feature_names="+" ".join(features)+"\n"+
                    "feature_infos="+" ".join(feature_infos)+"\n"+
                        # tree_sizes=324
                    "\n")

                    #feature_infos is minimum value to maximum value ratio of every feature
    
    model_file = get_tree_string(segmentation=segmentation_obj, temp_file = f)
    f.close()
    newgbm = lgb.basic.Booster(params = {'model_str' : open(filename, "r").read()})
    f.close()
#     newgbm1 = lgb.basic.Booster(model_file = filename)
    # os.remove(filename)
    return newgbm

