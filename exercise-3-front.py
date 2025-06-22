import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Recreate the ConditionalVAE class (needs to match the trained model)
class ConditionalVAE(keras.Model):
    """Conditional Variational Autoencoder for digit generation"""
    
    def __init__(self, latent_dim=20, num_classes=10, **kwargs):
        super(ConditionalVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
            layers.Flatten(),
            layers.Dense(16, activation="relu"),
        ])
        
        # Label embedding for conditioning
        self.label_embedding = layers.Embedding(num_classes, 50)
        
        # Mean and log variance for latent space
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim + 50,)),
            layers.Dense(7 * 7 * 64, activation="relu"),
            layers.Reshape((7, 7, 64)),
            layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same"),
        ])

    def encode(self, x, labels):
        encoded = self.encoder(x)
        label_emb = self.label_embedding(labels)
        label_emb = tf.reshape(label_emb, [-1, 50])
        combined = tf.concat([encoded, label_emb], axis=1)
        z_mean = self.z_mean(combined)
        z_log_var = self.z_log_var(combined)
        return z_mean, z_log_var

    def decode(self, z, labels):
        label_emb = self.label_embedding(labels)
        label_emb = tf.reshape(label_emb, [-1, 50])
        combined = tf.concat([z, label_emb], axis=1)
        return self.decoder(combined)
    
    def generate(self, labels, num_samples=1):
        """Generate new digit images for given labels"""
        # Sample from latent space
        z = tf.random.normal(shape=(num_samples, self.latent_dim))
        
        # Handle labels
        if isinstance(labels, int):
            labels = tf.constant([labels] * num_samples)
        elif len(labels.shape) == 0:
            labels = tf.repeat([labels], num_samples)
        
        # Generate images
        generated = self.decode(z, labels)
        return generated

@st.cache_resource
def load_trained_model():
    """Load the trained VAE model"""
    try:
        # Check if model files exist
        if not os.path.exists('digit_generator_model.h5'):
            return None, "Model not found. Please train the model first by running exercise-3-back.py"
        
        if not os.path.exists('model_config.pkl'):
            return None, "Model configuration not found."
        
        # Load model configuration
        with open('model_config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        # Create model instance
        model = ConditionalVAE(
            latent_dim=config['latent_dim'],
            num_classes=config['num_classes']
        )
        
        # Build the model by calling it with dummy data
        dummy_images = tf.random.normal((1, 28, 28, 1))
        dummy_labels = tf.constant([0])
        _ = model([dummy_images, dummy_labels])
        
        # Load weights
        model.load_weights('digit_generator_model.h5')
        
        return model, "Model loaded successfully!"
        
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def generate_digit_images(model, digit, num_examples=5, seed=None):
    """Generate images for a specific digit using the trained model"""
    if seed is not None:
        tf.random.set_seed(seed)
    
    # Generate images
    generated = model.generate(digit, num_samples=num_examples)
    
    # Convert to numpy and ensure proper range [0, 1]
    images = generated.numpy()
    images = np.clip(images, 0, 1)
    
    return images

def display_generated_images(images, digit):
    """Display generated images in a grid layout"""
    num_images = len(images)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    
    if num_images == 1:
        axes = [axes]
    
    for i, (ax, image) in enumerate(zip(axes, images)):
        # Squeeze to remove channel dimension for display
        img_display = image.squeeze()
        ax.imshow(img_display, cmap='gray')
        ax.set_title(f'Generated {i+1}')
        ax.axis('off')
    
    plt.suptitle(f'AI Generated Handwritten Digit: {digit}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def main():
    st.set_page_config(
        page_title="AI Digit Generator",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI Handwritten Digit Generator")
    st.markdown("### Enter a digit to see AI-generated handwritten examples")
    
    # Load the trained model
    with st.spinner("Loading AI model..."):
        model, status_message = load_trained_model()
    
    if model is None:
        st.error(status_message)
        st.markdown("""
        **To use this application:**
        1. First run the training script to create the AI model:
           ```bash
           source car_horn_env/bin/activate
           python exercise-3-back.py
           ```
        2. Wait for training to complete (this may take some time)
        3. Refresh this page to load the trained model
        """)
        return
    
    st.success(status_message)
    
    # User interface
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("#### Controls")
        
        digit = st.selectbox(
            "Select a digit (0-9):",
            options=list(range(10)),
            index=0,
            help="Choose which digit the AI should generate"
        )
        
        num_examples = st.slider(
            "Number of examples:",
            min_value=1,
            max_value=8,
            value=5,
            help="How many AI-generated examples to create"
        )
        
        # Seed for reproducibility
        use_seed = st.checkbox("Use random seed", help="For reproducible results")
        seed = None
        if use_seed:
            seed = st.number_input("Random seed:", min_value=0, max_value=9999, value=42)
        
        generate_button = st.button("🎨 Generate New Examples", help="Create new AI-generated digits")
    
    with col2:
        st.markdown(f"### AI Generated Digit **{digit}**")
        
        # Generate or regenerate images
        if generate_button or 'last_digit' not in st.session_state or st.session_state.last_digit != digit:
            with st.spinner("AI is creating handwritten digits..."):
                try:
                    # Generate images
                    generated_images = generate_digit_images(model, digit, num_examples, seed)
                    
                    # Store in session state
                    st.session_state.generated_images = generated_images
                    st.session_state.last_digit = digit
                    st.session_state.last_num_examples = num_examples
                    
                except Exception as e:
                    st.error(f"Error generating images: {str(e)}")
                    return
        
        # Display images if available
        if 'generated_images' in st.session_state:
            # Adjust if number of examples changed
            if len(st.session_state.generated_images) != num_examples:
                with st.spinner("Adjusting number of examples..."):
                    generated_images = generate_digit_images(model, digit, num_examples, seed)
                    st.session_state.generated_images = generated_images
            
            # Display the images
            fig = display_generated_images(st.session_state.generated_images, digit)
            st.pyplot(fig)
            
            # Show generation info
            st.markdown("---")
            st.markdown(f"""
            **AI Generation Info:**
            - Model Type: **Conditional Variational Autoencoder (VAE)**
            - Generated Images: **{len(st.session_state.generated_images)}**
            - Target Digit: **{digit}**
            - Image Resolution: **28x28 pixels**
            - Latent Dimensions: **20**
            """)
            
            # Add download option
            if st.button("💾 Save Generated Images"):
                for i, img in enumerate(st.session_state.generated_images):
                    img_pil = Image.fromarray((img.squeeze() * 255).astype(np.uint8))
                    img_pil.save(f"generated_digit_{digit}_example_{i+1}.png")
                st.success(f"Saved {len(st.session_state.generated_images)} images!")
        
        else:
            st.info("Click 'Generate New Examples' to create AI-generated digits!")
    
    # Additional information
    with st.expander("🧠 About the AI Model"):
        st.markdown("""
        This application uses a **Conditional Variational Autoencoder (VAE)** to generate handwritten digits.
        
        **How it works:**
        1. **Training**: The model learns from 60,000 real handwritten digits from the MNIST dataset
        2. **Encoding**: It learns to compress digit images into a compact representation
        3. **Generation**: It can create new digit images by sampling from this learned space
        4. **Conditioning**: The model is told which digit to generate, so it creates targeted examples
        
        **Model Architecture:**
        - **Encoder**: Convolutional layers that compress 28x28 images into 20-dimensional vectors
        - **Decoder**: Deconvolutional layers that reconstruct images from the compressed representation
        - **Conditional**: Uses digit labels to guide generation
        
        **Key Features:**
        - Generates unique, never-before-seen handwritten digits
        - Each generation is slightly different due to random sampling
        - Learned from real human handwriting patterns
        - Optimized for Raspberry Pi performance
        """)

if __name__ == "__main__":
    main()
