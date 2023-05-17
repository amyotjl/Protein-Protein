import onnx
from onnx.external_data_helper import load_external_data_for_model


# Méthode 1
# Tous les fichiers sont au même niveau dans un seul répertoire.
# Si on déplace .onnx et shared.weight dans /MatMul/ (ou déplacer tous les fichiers onnx_MatMul_xxxx vers onnx/), la commande suivant fonctionne.
# Exemple: 
# - onnx/
#   - MatMul/
#       - Roastlab.onnx
#       - shared.weight
#       - onnx_MatMul_2297
#       - onnx_MatMul_2313
#       ...
onnx_model1 = onnx.load("onnx/MatMul/Rostlab_prot_t5_xl_half_uniref50-enc.onnx")

# Méthode 2
# Le fichier .onnx n'est pas dans le même répertoire que le reste (onnx_MatMul_xxxx et shared.weight)
# Tous les fichiers onnx_MatMul_xxxx et shared.weight doivent être dans le même répertoire (j'ai déplacé shared.weight dans OneDrive)
# Exemple: 
# - onnx/
#   - Roastlab.onnx
#   - MatMul/
#       - shared.weight
#       - onnx_MatMul_2297
#       - onnx_MatMul_2313
#       ...
onnx_model2 = onnx.load("onnx/Rostlab_prot_t5_xl_half_uniref50-enc.onnx", load_external_data = False)
load_external_data_for_model(onnx_model2, 'onnx/MatMul')
