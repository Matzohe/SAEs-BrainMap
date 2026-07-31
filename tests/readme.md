### 相关的中间变量是通过哪些函数获得的，便于去寻找相关的文件

0、Voxel和SAE的特征之间的相关性保存代码以及路径:
函数名称：mean_similarity_analysis
函数路径：tests/sae/sae_brain_similarity/sae_brain_simlarity.py
效果：会将大脑voxel和SAEs特征的逐层相关性，保存在args.similarity.brain_sae_similarity_save_root中的路径下

1、roi和sae特征的逐层相关性提取
函数名称：get_target_roi_correlation
函数路径：tests/sae/sae_brain_similarity/brain_selected_sae.py  line 38
函数输入：args: EasyDict, roi_name: str, target_layer: int, subj: int
        需要注意的是，这里面会提取对应模型在NSD图像数据集上的中间激活，要分析的模型在args中，这个函数可以单独取出来运行

2、roi和sae特征的最相关的topk特征保存
函数名称：all_layer_feature_extraction
函数路径：tests/sae/sae_brain_similarity/brain_guide_circurt.py  line 16
函数输入：args: EasyDict, roi_name: str, subj: int, all_layers: int = 12, topk: int = 100,
        这个函数也可以单独拿出来运行

3、rsa计算代码：
函数名称：voxel_dictionary_rsa_selection
函数路径：src/sae_brain_correlation/brain_sae_rsa.py
