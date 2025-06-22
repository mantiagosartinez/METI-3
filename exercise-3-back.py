import kagglehub
import numpy as np
import struct
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("hojjatk/mnist-dataset")
print("Path to dataset files:", path)

def load_mnist_images(filename):
    """Load MNIST images from IDX file format"""
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        
    return images

def load_mnist_labels(filename):
    """Load MNIST labels from IDX file format"""
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels

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
            layers.InputLayer(input_shape=(latent_dim + 50,)),  # latent + label embedding
            layers.Dense(7 * 7 * 64, activation="relu"),
            layers.Reshape((7, 7, 64)),
            layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same"),
        ])

    def encode(self, x, labels):
        # Encode image
        encoded = self.encoder(x)
        
        # Embed labels
        label_emb = self.label_embedding(labels)
        label_emb = tf.reshape(label_emb, [-1, 50])
        
        # Combine encoded image with label embedding
        combined = tf.concat([encoded, label_emb], axis=1)
        
        z_mean = self.z_mean(combined)
        z_log_var = self.z_log_var(combined)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def decode(self, z, labels):
        # Embed labels
        label_emb = self.label_embedding(labels)
        label_emb = tf.reshape(label_emb, [-1, 50])
        
        # Combine latent vector with label embedding
        combined = tf.concat([z, label_emb], axis=1)
        
        return self.decoder(combined)

    def call(self, inputs):
        x, labels = inputs
        z_mean, z_log_var = self.encode(x, labels)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z, labels)
        
        # Add KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        
        return reconstructed
    
    def generate(self, labels, num_samples=1):
        """Generate new digit images for given labels"""
        # Sample from latent space
        z = tf.random.normal(shape=(num_samples, self.latent_dim))
        
        # Repeat labels if needed
        if len(labels.shape) == 0:  # Single label
            labels = tf.repeat([labels], num_samples)
        
        # Generate images
        generated = self.decode(z, labels)
        return generated

def prepare_data():
    """Load and prepare MNIST data"""
    print("Loading MNIST data...")
    
    # Try to find the dataset files
    train_images_path = None
    train_labels_path = None
    
    # Check different possible paths
    possible_paths = [
        # Kagglehub download paths - try directory first, then direct file
        (f"{path}/train-images-idx3-ubyte/train-images-idx3-ubyte", f"{path}/train-labels-idx1-ubyte/train-labels-idx1-ubyte"),
        (f"{path}/train-images-idx3-ubyte", f"{path}/train-labels-idx1-ubyte"),
        (f"{path}/train-images.idx3-ubyte", f"{path}/train-labels.idx1-ubyte"),
        # Local archive paths
        ("archive/train-images-idx3-ubyte/train-images-idx3-ubyte", "archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte"),
        ("archive/train-images.idx3-ubyte", "archive/train-labels.idx1-ubyte"),
        # Legacy paths
        ("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte")
    ]
    for img_path, lbl_path in possible_paths:
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            train_images_path = img_path
            train_labels_path = lbl_path
            break
    
    if not train_images_path:
        raise FileNotFoundError("Could not find MNIST dataset files")
    
    # Load data
    images = load_mnist_images(train_images_path)
    labels = load_mnist_labels(train_labels_path)
    
    # Normalize and reshape
    images = images.astype('float32') / 255.0
    images = np.expand_dims(images, -1)  # Add channel dimension
    
    print(f"Loaded {len(images)} images")
    return images, labels

def train_model():
    """Train the conditional VAE model"""
    print("Preparing data...")
    images, labels = prepare_data()
    
    # Create and compile model
    print("Creating model...")
    vae = ConditionalVAE(latent_dim=20, num_classes=10)
    vae.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Train the model
    print("Training model...")
    history = vae.fit(
        [images, labels], images,
        epochs=20,  # Reduced for Raspberry Pi
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    
    # Save the model
    print("Saving model...")
    vae.save_weights('digit_generator_model.h5')
    
    # Save model configuration
    model_config = {
        'latent_dim': 20,
        'num_classes': 10,
        'input_shape': (28, 28, 1)
    }
    
    with open('model_config.pkl', 'wb') as f:
        pickle.dump(model_config, f)
    
    print("Model training completed!")
    
    # Test generation
    print("Testing model generation...")
    for digit in range(10):
        generated = vae.generate(tf.constant([digit]), num_samples=1)
        generated_image = generated[0].numpy().squeeze()
        
        plt.figure(figsize=(2, 2))
        plt.imshow(generated_image, cmap='gray')
        plt.title(f'Generated {digit}')
        plt.axis('off')
        plt.savefig(f'test_generated_{digit}.png')
        plt.close()
    
    print("Test images saved!")
    return vae

if __name__ == "__main__":
    # Train the model
    model = train_model()
    print("Training complete! Model saved as 'digit_generator_model.h5'")