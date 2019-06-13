from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)


from pprint import pprint
from PMML44 import *
from datetime import datetime
import nyoka.metadata as md
import nyoka.xgboost.xgboost_to_pmml as xgboostToPmml

class Export:
    def __init__(self, model = None, col_names = None, target_name = None, pmml_f_name='from_lgbm.pmml'):

        def ExportToPMML(model=None):
            jsondata = model.dump_model()
            feature_names = jsondata['feature_names']
            objective = jsondata['objective']
            tree_info = jsondata['tree_info']

            pmml = PMML(
                version = '4.4',
                Header = Header(copyright = "Copyright (c) 2018 Software AG", description = "PMML from LGBM Training API",  
                                Timestamp = Timestamp(datetime.utcnow()),
                                Application = Application(name = "Nyoka", version = md.__version__)),
                                DataDictionary= DataDictionary(),
                                MiningModel = [get_MiningModel(tree_information=tree_info, features = feature_names)])
            pmml.export(outfile = open(pmml_f_name, "w"), level = 0)

        def get_MiningModel(tree_information = None, features = None):
            #MiningSc = MiningSchema(MiningField=[MiningField(name = feature, lowValue = data[feature].min(), highValue = data[feature].max() ) for feature in features])
            MiningSc = MiningSchema(MiningField=[MiningField(name = feature) for feature in features])
            op = Output()
            seg = get_Segmentation(tree_information=tree_information, features = features)
            mm = MiningModel(modelName="LightGBModel", functionName = "regression", MiningSchema=MiningSc, Output=op, Segmentation= seg)
            return mm
        
        def get_Segmentation(tree_information=None, features = None):
            segmentation_list = list()
            for i in range(len(tree_information)):
                tree_data = tree_information[i]
                tree = get_TreeModel(treeData=tree_data, features = features)
                segmentation_list.append(Segment(id=i+1, TreeModel=tree))
            segmentation = Segmentation(Segment=segmentation_list)
            return segmentation

        def get_TreeModel(treeData = None, features = None):
            #ms = MiningSchema(MiningField=[MiningField(name = feature, lowValue = data[feature].min(), highValue = data[feature].max() ) for feature in features])
            ms = MiningSchema(MiningField=[MiningField(name = feature) for feature in features])
            node = get_Tree(treeData=treeData['tree_structure'],features = features)
            tree = TreeModel(modelName="DecisionTreeModel", functionName="regression", MiningSchema=ms, Node=node)
            tree.set_Extension([Extension(name='shrinkage', value=treeData['shrinkage'])])
            return tree

        def get_Tree(treeData = None, features = None):
            rootNode = Node()
            create_node(treeData,rootNode,features)
            return rootNode

        def create_node(obj, main_node,derived_col_names):

            def create_left_node(obj,derived_col_names):
                nd = Node()
                nd.set_SimplePredicate(
                    SimplePredicate(field=derived_col_names[int(obj['split_feature'])], operator='lessOrEqual', value=obj['threshold']))
                create_node(obj['left_child'], nd, derived_col_names)
                return nd

            def create_right_node(obj,derived_col_names):
                nd = Node()
                nd.set_SimplePredicate(
                    SimplePredicate(field=derived_col_names[int(obj['split_feature'])], operator='greaterThan', value=obj['threshold']))
                create_node(obj['right_child'], nd, derived_col_names)
                return nd

            if 'leaf_index' in obj:
                main_node.set_score(obj['leaf_value'])
                main_node.set_recordCount(obj['leaf_count'])
                main_node.set_id(obj['leaf_index'])
            elif 'split_index' in obj:
                main_node.set_score(obj['internal_value'])
                main_node.set_recordCount(obj['internal_count'])
                main_node.set_id(obj['split_index'])
                main_node.set_Extension([Extension(name='gain', value=obj['split_gain'])])
                main_node.add_Node(create_left_node(obj,derived_col_names))
                main_node.add_Node(create_right_node(obj,derived_col_names))
				


        ExportToPMML(model=model)