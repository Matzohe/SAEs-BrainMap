# 1. All project paths required by SAEs-BrainMap are configured in `config.toml`.
#
# 2. To avoid repeated inference on the ImageNet test set, this project
#    caches the intermediate activations of every model at every layer
#    for all test images. Since these cached activations can occupy a
#    large amount of storage, it is recommended to set the activation
#    cache directory to a high-capacity storage device (e.g., an HDD or
#    large external drive)

# For easy reproduct, we released several results including
# - The SAEs weight
# - The brain-saes activation similarity matrix (subj5)
# - The ROI mask

# Training SAEs for different models, support different models
# including clip_vit-b_16, imagenet, dinov2, mae
# We provided the SAEs ckpt in huggingface
nohup python tests/test/sae_training.py --exp_subj 5 --exp_model_name "clip_vit-b_16" --exp_device "cuda:0" --autoencoder_name "original" --autoencoder_rate 16

# Evaluate Brain-SAE activation similarity
# This command calcuate the neuron and SAE's similarity with Brain
# This function need the original NSD fMRI activation, we provide the voxel-SAEs activation similarity matrix in huggingface.
python tests/test/brain_model_sae_similarity.py --exp_subj 5 --exp_model_name "clip_vit-b_16" --exp_device "cuda:0" --autoencoder_name "original" --autoencoder_rate 16

# Computing representation similarity score
# This function need the original NSD fMRI activation.
python tests/test/brain_sae_rsa_analysis.py --exp_subj 5 --exp_model_name "clip_vit-b_16" --exp_device "cuda:0" --autoencoder_name "original" --autoencoder_rate 16

# Before analysis the SAEs Feature, save the middle activation of SAEs on Imagenet test
# WARNING: Requires A large amount of storage, change the root in config first: SAEsEvaluation.imagenet_test_token_save_root
python imagenet_sae_extraction.py

# ROI based SAEs layerwise feature selection based on the brain-sae activation similarity
# First, selected the Features that most correlated with the target ROI
# Then, read all it's activation on ImageNet test
# Finally, select the top 20 activated figure and visualize the activation heatmap
# the main function is selected_sae_feature_activation_analysis in tests/sae/sae_brain_similarity/brain_guide_circuit.py
# WARNING: the image save root in config is args.similarity.roi_selected_feature_heatmap_independent_save_root
# Need large storage
python tests/test/selected_feature_visualize.py --exp_subj 5 --exp_model_name "clip_vit-b_16" --exp_device "cuda:0" --autoencoder_name "original" --autoencoder_rate 16

# The results will be best after filtered by CLIP
# Clip based feature stability and functional aligned evaluation
# After that, save the Selected SAEs feature visualize rusults
# Then we can get the main results.
python clip_evaluation_and_visualization.py

# Due to the complexity of the codebase, some legacy functions and experimental implementations 
# have been retained but are no longer used. If you encounter any confusion or have any questions, 
# please feel free to contact me at: maoziming@westlake.edu.cn