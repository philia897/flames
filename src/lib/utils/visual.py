from .dbhandler import JsonDBHandler, Items
import graphviz

def id_trans(*args):  # Cut the first 8 digits of id
    st = list()
    for arg in args:
        st.append(arg[:8])
    return st

def generate_graph(handler:JsonDBHandler, save_dir_path=None):
    all_nodes:dict = handler.read_all(Items.MODEL_INFO)
    graph = graphviz.Digraph('model-graph')
    for k in all_nodes.keys():
        graph.node(*id_trans(k))  
    for k,v in all_nodes.items():
        for c in v['children']:
            graph.edge(*id_trans(k, c))
        if len(v['children']) == 0:
            graph.node(*id_trans(k), color='lightgrey', style='filled')
    graph.render(directory=save_dir_path)
    return graph