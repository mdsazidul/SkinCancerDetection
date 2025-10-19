import streamlit as st
import torch
from torch import nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download


# CONFIGURATION

HF_MODEL_REPO_ID = "sazidshovon/SkinCancerDetection"
MODEL_FILENAME = "best_model.pth"
IMAGE_SIZE = 384
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class information
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

CLASS_INFO = {
    'akiec': {
        'name': 'Actinic Keratoses',
        'risk': 'Precancerous',
        'color': '#FFA500',
        'description': 'Precancerous lesions caused by sun damage. Requires medical evaluation.',
        'action': '⚠️ Consult dermatologist for treatment options'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'risk': 'Malignant',
        'color': '#FF6347',
        'description': 'Most common skin cancer. Highly treatable when caught early.',
        'action': '🚨 URGENT: Schedule dermatologist appointment immediately'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'risk': 'Benign',
        'color': '#90EE90',
        'description': 'Non-cancerous skin growth. Generally harmless.',
        'action': '✅ Monitor for changes, routine checkup recommended'
    },
    'df': {
        'name': 'Dermatofibroma',
        'risk': 'Benign',
        'color': '#90EE90',
        'description': 'Benign fibrous nodule. Usually harmless.',
        'action': '✅ No immediate action needed, monitor for changes'
    },
    'mel': {
        'name': 'Melanoma',
        'risk': 'Malignant',
        'color': '#DC143C',
        'description': 'Most dangerous skin cancer. Requires immediate attention.',
        'action': '🚨 CRITICAL: Seek immediate medical attention'
    },
    'nv': {
        'name': 'Melanocytic Nevus',
        'risk': 'Benign',
        'color': '#90EE90',
        'description': 'Common mole. Usually benign but monitor for changes.',
        'action': '✅ Regular self-examination recommended'
    },
    'vasc': {
        'name': 'Vascular Lesion',
        'risk': 'Benign',
        'color': '#90EE90',
        'description': 'Blood vessel abnormality. Typically benign.',
        'action': '✅ Routine evaluation recommended'
    }
}

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Skin Cancer Detection AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #FF3B3B 0%, #FF5B5B 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    .result-card {
        padding: 2rem;
        border-radius: 1rem;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #FFF3CD;
        border-left: 5px solid #FFA500;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #FFE5E5;
        border-left: 5px solid #DC143C;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #E5F5E5;
        border-left: 5px solid #90EE90;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================
# MODEL DEFINITION
# ===========================
class ModelWithUncertainty(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, x):
        return self.base_model(x)
    
    def predict_with_uncertainty(self, x, num_samples=15):
        """Monte Carlo Dropout for uncertainty estimation"""
        self.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.base_model(x)
                predictions.append(torch.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty

# ===========================
# MODEL LOADING
# ===========================
@st.cache_resource
def load_model():
    """Load model from Hugging Face Hub"""
    with st.spinner('🔄 Loading AI model from Hugging Face...'):
        try:
            # Download model from HF Hub
            model_path = hf_hub_download(
                repo_id=HF_MODEL_REPO_ID,
                filename=MODEL_FILENAME,
                repo_type="model"
            )
            
            # Create base model
            base_model = timm.create_model(
                'swin_base_patch4_window12_384',
                pretrained=False,
                num_classes=NUM_CLASSES,
                drop_rate=0.3,
                drop_path_rate=0.2
            )
            
            # Wrap with uncertainty
            model = ModelWithUncertainty(base_model)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()
            
            return model
            
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            st.info(f"Please verify:\n- Repo: {HF_MODEL_REPO_ID}\n- File: {MODEL_FILENAME}")
            st.stop()

# ===========================
# IMAGE PREPROCESSING
# ===========================
def preprocess_image(image):
    """
    CRITICAL: Use EXACT same normalization as training!
    Training used: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)

# ===========================
# IMAGE VALIDATION
# ===========================
def validate_skin_image(image):
    """Check if image appears to be a skin lesion"""
    img_array = np.array(image.resize((224, 224)))
    
    # Convert to HSV
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Detect skin tones (broad range)
    lower_skin = np.array([0, 15, 60], dtype=np.uint8)
    upper_skin = np.array([30, 255, 255], dtype=np.uint8)
    
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
    
    # Check texture variance
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    variance = np.var(gray) / 255.0
    
    # Image should have skin tones and texture
    is_valid = skin_ratio > 0.25 and variance > 0.008
    
    return is_valid, skin_ratio, variance

# ===========================
# GRAD-CAM VISUALIZATION
# ===========================
def generate_gradcam(img_tensor, model, class_idx):
    """Generate Grad-CAM heatmap"""
    model.eval()
    gradients = []
    activations = []
    
    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def save_activations(module, input, output):
        activations.append(output)
    
    # Hook into final layer
    target_layer = model.base_model.head
    handle_acts = target_layer.register_forward_hook(save_activations)
    handle_grads = target_layer.register_full_backward_hook(save_gradients)
    
    img_tensor = img_tensor.to(DEVICE)
    img_tensor.requires_grad = True
    
    output = model(img_tensor)
    model.zero_grad()
    
    class_loss = output[0, class_idx]
    class_loss.backward()
    
    handle_acts.remove()
    handle_grads.remove()
    
    if len(gradients) > 0 and len(activations) > 0:
        grads = gradients[0].cpu().detach().numpy()[0]
        acts = activations[0].cpu().detach().numpy()[0]
        
        # Handle different tensor shapes
        if len(grads.shape) > 1:
            weights = np.mean(grads, axis=tuple(range(1, len(grads.shape))))
            cam = np.zeros(acts.shape[1:] if len(acts.shape) > 2 else acts.shape, dtype=np.float32)
            
            for i, w in enumerate(weights):
                if len(acts.shape) > 2:
                    cam += w * acts[i]
            
            cam = np.maximum(cam, 0)
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            
            cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
            return cam
    
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE))

def overlay_heatmap(image, cam):
    """Create heatmap overlay on original image"""
    img_array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = img_array * 0.5 + heatmap * 0.5
    return Image.fromarray(np.uint8(overlay))


# PREDICTION FUNCTION

def predict_image(image, model):
    """Make prediction with uncertainty estimation"""
    # Preprocess
    img_tensor = preprocess_image(image)
    img_tensor = img_tensor.to(DEVICE)
    
    # Get prediction with uncertainty
    mean_probs, uncertainty = model.predict_with_uncertainty(img_tensor, num_samples=15)
    
    # Extract results
    mean_probs = mean_probs[0].cpu().numpy()
    uncertainty = uncertainty[0].cpu().numpy()
    
    pred_class = int(np.argmax(mean_probs))
    confidence = float(mean_probs[pred_class])
    max_uncertainty = float(uncertainty[pred_class])
    
    return pred_class, confidence, max_uncertainty, mean_probs


# VISUALIZATION

def create_probability_chart(probabilities):
    """Create interactive probability bar chart"""
    colors = [CLASS_INFO[name]['color'] for name in CLASS_NAMES]
    
    fig = go.Figure(data=[
        go.Bar(
            x=CLASS_NAMES,
            y=probabilities * 100,
            marker=dict(color=colors),
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Condition",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 110],
        height=400,
        showlegend=False,
        hovermode='x'
    )
    
    return fig


# MAIN APPLICATION

def main():
    # Header
    st.title("🔬 Advanced Skin Lesion Detection System")
    st.markdown("### AI-Powered Dermatology Assistant with Uncertainty Estimation")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(f"""
        **Model**: Swin Transformer Base  
        **Classes**: {NUM_CLASSES} skin conditions  
        **Device**: {DEVICE}
        
        This system uses deep learning to analyze dermatoscopic images 
        and classify skin lesions with uncertainty quantification.
        """)
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        confidence_threshold = st.slider(
            "Minimum Confidence (%)",
            min_value=50,
            max_value=95,
            value=70,
            step=5,
            help="Predictions below this threshold will show a warning"
        )
        
        uncertainty_threshold = st.slider(
            "Maximum Uncertainty (%)",
            min_value=5,
            max_value=30,
            value=15,
            step=1,
            help="Predictions above this uncertainty will show a warning"
        )
        
        show_gradcam = st.checkbox("Show Attention Heatmap", value=True)
        validate_image = st.checkbox("Validate Image Quality", value=True)
        
        st.markdown("---")
        st.warning("""
        ⚠️ **Medical Disclaimer**  
        This is a screening tool, NOT a diagnostic device. 
        Always consult a dermatologist for proper medical evaluation.
        """)
    
    # Load model
    try:
        model = load_model()
        st.success(f"✅ Model loaded successfully on {DEVICE}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a dermatoscopic image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear, close-up image of the skin lesion"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze Lesion", type="primary"):
                with st.spinner("🔬 Analyzing image..."):
                    
                    # Validate image
                    if validate_image:
                        is_valid, skin_ratio, variance = validate_skin_image(image)
                        
                        if not is_valid:
                            st.markdown(f"""
                            <div class="warning-box">
                            <h4>⚠️ Image Quality Warning</h4>
                            <p>This image may not be a proper dermatoscopic image:</p>
                            <ul>
                                <li>Skin tone ratio: {skin_ratio*100:.1f}% (expected >25%)</li>
                                <li>Texture variance: {variance:.3f} (expected >0.008)</li>
                            </ul>
                            <p>For best results, upload a close-up photo of a skin lesion.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    try:
                        # Predict
                        pred_class, confidence, uncertainty, probabilities = predict_image(image, model)
                        
                        # Generate Grad-CAM
                        if show_gradcam:
                            img_tensor = preprocess_image(image)
                            cam = generate_gradcam(img_tensor, model, pred_class)
                            heatmap = overlay_heatmap(image, cam)
                        else:
                            heatmap = None
                        
                        # Store in session state
                        st.session_state.prediction = {
                            'class': pred_class,
                            'confidence': confidence,
                            'uncertainty': uncertainty,
                            'probabilities': probabilities,
                            'heatmap': heatmap
                        }
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                        st.exception(e)
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'prediction' in st.session_state:
            pred = st.session_state.prediction
            pred_class_name = CLASS_NAMES[pred['class']]
            info = CLASS_INFO[pred_class_name]
            
            # Reliability check
            is_reliable = (
                pred['confidence'] >= confidence_threshold/100 and
                pred['uncertainty'] <= uncertainty_threshold/100
            )
            
            # Main prediction card
            if info['risk'] == 'Malignant':
                box_class = 'error-box'
            elif info['risk'] == 'Precancerous':
                box_class = 'warning-box'
            else:
                box_class = 'success-box'
            
            st.markdown(f"""
            <div class="{box_class}">
                <h2>🎯 Predicted Condition</h2>
                <h1 style="color: {info['color']};">{info['name']}</h1>
                <h3>Class: {pred_class_name.upper()}</h3>
                <p><strong>Confidence:</strong> {pred['confidence']*100:.2f}%</p>
                <p><strong>Uncertainty:</strong> ±{pred['uncertainty']*100:.2f}%</p>
                <p><strong>Risk Level:</strong> {info['risk']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Reliability warning
            if not is_reliable:
                if pred['confidence'] < confidence_threshold/100:
                    st.warning(f"⚠️ Low confidence ({pred['confidence']*100:.1f}%). Results may be unreliable.")
                if pred['uncertainty'] > uncertainty_threshold/100:
                    st.warning(f"⚠️ High uncertainty (±{pred['uncertainty']*100:.1f}%). Consider expert consultation.")
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Confidence", f"{pred['confidence']*100:.1f}%")
            with col_b:
                st.metric("Uncertainty", f"±{pred['uncertainty']*100:.1f}%")
            with col_c:
                st.metric("Risk", info['risk'])
            
            # Description and action
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Recommended Action:** {info['action']}")
            
            # Probability chart
            st.plotly_chart(
                create_probability_chart(pred['probabilities']),
                use_container_width=True
            )
            
            # Detailed probabilities
            with st.expander("📋 Detailed Class Probabilities"):
                for i, (name, prob) in enumerate(zip(CLASS_NAMES, pred['probabilities'])):
                    class_info = CLASS_INFO[name]
                    st.markdown(f"""
                    **{class_info['name']}** (`{name}`)  
                    Risk: {class_info['risk']} | Probability: **{prob*100:.2f}%**
                    """)
                    st.progress(float(prob))
            
            # Grad-CAM
            if pred['heatmap'] is not None:
                st.markdown("### 🔥 Attention Heatmap (Grad-CAM)")
                st.image(pred['heatmap'], caption="Model Attention Areas", use_container_width=True)
                st.info("Red/yellow areas indicate where the model focused its attention during analysis.")
            
            # Reset button
            if st.button("🔄 Analyze Another Image"):
                del st.session_state.prediction
                st.rerun()
        else:
            st.info("👆 Upload an image and click 'Analyze Lesion' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>Developed by MD Sazidul Islam & Dr. Shakil Akhtar</strong></p>
        <p>Website: <a href='https://www.mdsazidulislam.site' target='_blank'>www.mdsazidulislam.site</a></p>
        <p>⚠️ This tool is for educational/screening purposes only - Not for clinical diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
