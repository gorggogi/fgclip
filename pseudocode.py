def project_embeddings(input_vector):
    """
    Projects a high-dimensional feature vector (768D) into a 
    lower-dimensional latent space (512D) to match the other modality.
    """

    # 1. 'Before' State: Raw Output from Text Transformer
    # The vector has 768 features (standard for BERT-based models).
    # Shape: (1, 768)
    input_768 = input_vector 

    # 2. Define Projection Weights (The Learned Linear Layer)
    # A matrix of trainable weights connecting every input neuron to every output neuron.
    # Matrix Shape: (768, 512)
    weight_matrix = get_learned_weights() 
    bias_vector = get_learned_bias()      # Shape: (512,)

    # 3. Linear Transformation (Matrix Multiplication)
    # Logic: (1, 768) dot (768, 512) = (1, 512)
    # This compresses the information while preserving semantic meaning.
    output_512 = (input_768 @ weight_matrix) + bias_vector

    # 4. 'After' State: Projected Embedding
    # The vector is now compatible with the 512D image vectors.
    return output_512