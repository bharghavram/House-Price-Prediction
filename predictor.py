#prediction using gradientboosting
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, RobustScaler
import pickle
import warnings
import streamlit as st
from PIL import Image
import tempfile
import joblib
warnings.filterwarnings('ignore')

# Configuration
IMG_SIZE = 229
MODEL_PATH = "path of\\gradient_boosting_model.joblib"
PRICE_LOC_ENCODER="path of\\location_encoder.joblib"
AREA_SCALER="path of\price prediction\\area_scaler .joblib"
PRICE_SCALER="path of\\price prediction\\price_scaler.joblib"
# Location prediction
LOC_MODEL_PATH = "path of\\location_prediction\\location_model_complete.h5"
LOC_ENCODER_PATH = "path of\\location_prediction\\location_label_encoder.pkl"

# Feature names used during training

@st.cache_resource
def load_models_and_preprocessors():
    """Load all models and preprocessors with caching"""
    try:
        # Load price prediction model
        price_model = joblib.load(MODEL_PATH)
        print("price_model was sucessfully loaded")
        
        # Load location prediction model
        location_model = load_model(LOC_MODEL_PATH)
        print("location_model was sucessfully loaded")
        
        # Load price prediction preprocessors
        area_scaler=joblib.load(AREA_SCALER)
        print("area_scaler was sucessfully loaded")
        price_loc_encoder=joblib.load(PRICE_LOC_ENCODER)
        print("price_loca _encoder was sucessfully loaded")
        price_scaler=joblib.load(PRICE_SCALER)
        print("price_scaler was sucessfully loaded")
        
        # Load location prediction encoder
        with open(LOC_ENCODER_PATH, 'rb') as f:
            location_encoder = pickle.load(f)
        print("loc_encoder was sucessfully loaded")
        return price_model, location_model, area_scaler, price_loc_encoder, price_scaler,location_encoder
    
    except Exception as e:
        st.error(f"Error loading models or preprocessors: {e}")
        return None, None, None, None, None

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None


def preprocess_image_for_location(img_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """Advanced image preprocessing with augmentation capabilities"""
    try:
        if not os.path.exists(img_path):
            return None

        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)

        # EfficientNet preprocessing
        #img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        print(img_array.shape)
        return img_array
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return None

    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None






def predict_location(img_path, location_model, location_encoder):
    """Predict location from image"""
    try:
        # Preprocess image
        img_array = preprocess_image_for_location(img_path)
        if img_array is None:
            return None
        
        # Make prediction
        with st.spinner('Predicting location from image...'):
            prediction = location_model.predict(img_array, verbose=0)
        
        # Get predicted class
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        
        # Convert back to location name
        predicted_location = location_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Get confidence score
        confidence = np.max(prediction) * 100
        
        return predicted_location, confidence
        
    except Exception as e:
        st.error(f"Error predicting location: {e}")
        return None, None
def predict_price(price_model, area_scaler, price_loc_encoder, price_scaler, price_data):
    """Predict price for sample data"""
    try:
        # Transform location
        loc = price_loc_encoder.transform([[price_data['location']]])
        
        # Transform area
        area = area_scaler.transform([[price_data['total_sqft']]])
        
        # Prepare other features
        bhk = [price_data['bhk']]
        balcony = [price_data['balcony']]
        bath = [price_data['bath']]

        
        # Combine features into a single array for prediction
        features = [[loc[0],bhk[0], area[0][0], bath[0], balcony[0], ]]
        print(features)
        # Make prediction
        res = price_model.predict(features)
        
        # Handle the scalar result properly
        if hasattr(res, 'shape') and len(res.shape) > 0:
            # If res is an array, get the first element
            predicted_scaled = res[0] if len(res.shape) == 1 else res[0][0]
        else:
            # If res is already a scalar
            predicted_scaled = res
        
        # Inverse transform - need to reshape to 2D array for sklearn scalers
        predicted_price = price_scaler.inverse_transform([[predicted_scaled]])[0][0]
        print(predicted_price)
        
        # Calculate price range (you can adjust this logic based on your needs)
        # Adding ±20% range as an example
        price_range_min = predicted_price * 0.8
        price_range_max = predicted_price * 1.2
        
        return {
            'predicted_price': predicted_price,
            'price_range_min': price_range_min,
            'price_range_max': price_range_max
        }
        
    except Exception as e:
        st.error(f"Error predicting price: {e}")
        print(f"Debug info - Error details: {str(e)}")  # For debugging
        return None

def main():
    """Main Streamlit application"""
    # Set page config
    st.set_page_config(
        page_title="🏠 Property Price Predictor", 
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Load models and preprocessors
    price_model, location_model, area_scaler, price_loc_encoder, price_scaler,location_encoder = load_models_and_preprocessors()
    
    if any(component is None for component in [price_model, location_model, area_scaler, price_loc_encoder, price_scaler,location_encoder]):
        st.error("❌ Failed to load models or preprocessors. Please check file paths.")
        st.info("Make sure all model files are in the correct directories:")
        st.code(f"Price Model: {MODEL_PATH}")
        st.code(f"Location Model: {LOC_MODEL_PATH}")
        st.code(f"Preprocessors: {AREA_SCALER,PRICE_LOC_ENCODER,PRICE_SCALER,LOC_ENCODER_PATH}")
        st.code(f"Location Encoder: {LOC_ENCODER_PATH}")
        return
    
    # UI Layout
    st.title("🏡 Bangalore Property Price Estimator")
    st.caption("AI-powered property valuation using satellite imagery and property features")
    st.divider()
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Property Image")
        # Image upload
        image_file = st.file_uploader(
            "Upload satellite image of the property", 
            type=["jpg", "jpeg", "png"],
            help="Upload a satellite/aerial view image of the property"
        )
        
        if image_file is not None:
            # Display uploaded image
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Property Image", use_column_width=True)
    
    with col2:
        st.subheader("🏠 Property Details")
        # Property details
        bhk = st.number_input("🛏️ BHK (Bedrooms)", min_value=1, max_value=10, value=2, help="Number of bedrooms")
        sqft = st.number_input("📐 Total Area (sqft)", min_value=100, max_value=10000, value=1000, help="Total area in square feet")
        bath = st.number_input("🛁 Bathrooms", min_value=1, max_value=6, value=2, help="Number of bathrooms")
        balcony = st.number_input("🌇 Balconies", min_value=0, max_value=5, value=1, help="Number of balconies")
    
    st.divider()
    
    # Predict Button
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        if image_file is None:
            st.warning("⚠️ Please upload a satellite image to proceed.")
        else:
            try:
                # Save uploaded file temporarily
                temp_image_path = save_uploaded_file(image_file)
                
                if temp_image_path is None:
                    st.error("❌ Failed to save uploaded image")
                    return
                
                # Step 1: Predict location from image
                st.info("🔍 Step 1: Analyzing image to predict location...")
                location_result = predict_location(temp_image_path, location_model, location_encoder)
                
                if location_result is None or location_result[0] is None:
                    st.error("❌ Failed to predict location from image")
                    return
                
                predicted_location, location_confidence = location_result
                
                # Display location prediction
                st.success(f"📍 **Predicted Location:** {predicted_location}")
                st.info(f"🎯 **Location Confidence:** {location_confidence:.1f}%")
                
                # Step 2: Predict price
                st.info("💰 Step 2: Calculating property price...")
                
                # Prepare data for price prediction
                price_data = {
                    'total_sqft': sqft,
                    'bhk': bhk,
                    'bath': bath,
                    'balcony': balcony,
                    'location': predicted_location
                }
                
                # Make price prediction
                price_result = predict_price(price_model, area_scaler,price_loc_encoder,price_scaler, price_data)
                
                # Clean up temporary file
                try:
                    os.unlink(temp_image_path)
                except:
                    pass  # Ignore cleanup errors
                
                if price_result is not None:
                    # Display results
                    st.success("✅ Prediction Complete!")
                    st.balloons()
                    
                    # Create result display
                    st.subheader("💎 Price Estimation Results")
                    
                    # Main price display
                    price_col1, price_col2, price_col3 = st.columns(3)
                    
                    with price_col1:
                        st.metric(
                            label="💰 Estimated Price",
                            value=f"₹{price_result['predicted_price']:,.0f}L",
                            help="Predicted property price in Lakhs"
                        )
                    
                    with price_col2:
                        st.metric(
                            label="📉 Lower Bound",
                            value=f"₹{price_result['price_range_min']:,.0f}L",
                            help="Lower estimate of property price"
                        )
                    
                    with price_col3:
                        st.metric(
                            label="📈 Upper Bound",
                            value=f"₹{price_result['price_range_max']:,.0f}L",
                            help="Upper estimate of property price"
                        )
                    
                    # Price range info
                    price_range = price_result['price_range_max'] - price_result['price_range_min']
                    st.info(f"💡 **Estimated Price Range:** ₹{price_result['price_range_min']:,.0f}L - ₹{price_result['price_range_max']:,.0f}L (Range: ₹{price_range:,.0f}L)")
                    
                    # Summary section
                    st.subheader("📋 Prediction Summary")
                    
                    summary_col1, summary_col2 = st.columns(2)
                    
                    with summary_col1:
                        st.write("**🏠 Property Features:**")
                        st.write(f"• **Location:** {predicted_location}")
                        st.write(f"• **BHK:** {bhk} bedrooms")
                        st.write(f"• **Area:** {sqft:,} sqft")
                        st.write(f"• **Bathrooms:** {bath}")
                        st.write(f"• **Balconies:** {balcony}")
                    
                    with summary_col2:
                        st.write("**🎯 Model Insights:**")
                        st.write(f"• **Location Confidence:** {location_confidence:.1f}%")
                        st.write(f"• **Price per sqft:** ₹{(price_result['predicted_price']*100000/sqft):,.0f}")
                        st.write(f"• **Prediction Range:** ±{(price_range/price_result['predicted_price']*100):.1f}%")
                        
                        # Add quality indicator
                        if location_confidence > 80:
                            st.write("• **Prediction Quality:** 🟢 High")
                        elif location_confidence > 60:
                            st.write("• **Prediction Quality:** 🟡 Medium")
                        else:
                            st.write("• **Prediction Quality:** 🔴 Low")
                
                else:
                    st.error("❌ Price prediction failed. Please try again with different inputs.")
                    
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.info("Please try again or contact support if the issue persists.")

    # Add footer
    st.divider()
    st.caption("🤖 Powered by AI | Built with TensorFlow & Streamlit")

if __name__ == "__main__":
    main()
