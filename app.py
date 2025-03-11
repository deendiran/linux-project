# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
import geopy
from geopy.geocoders import Nominatim
import json
import ee

# Initialize Earth Engine
try:
    ee.Initialize(project='ee-ndirangudenise61')
except:
    ee.Authenticate()
    ee.Initialize(project='ee-ndirangudenise61')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route("/api/endpoint", methods=["GET", "POST"])
def home():
    return jsonify({"message": "Hello from the backend!"})

# Add this line to make Flask work as a serverless function
def handler(event, context):
    from flask_lambda import FlaskLambda
    lambda_app = FlaskLambda(app)
    return lambda_app(event, context)

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'flood_prediction_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '1234'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

def init_db():
    """Initialize database tables"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id SERIAL PRIMARY KEY,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                location_name TEXT,
                risk_level VARCHAR(50),
                ndvi DOUBLE PRECISION,
                ndwi DOUBLE PRECISION,
                soil_moisture DOUBLE PRECISION,
                water_level DOUBLE PRECISION,
                confidence_score DOUBLE PRECISION,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS historical_data (
                id SERIAL PRIMARY KEY,
                analysis_id INTEGER REFERENCES analysis_results(id),
                date DATE,
                ndvi DOUBLE PRECISION,
                ndwi DOUBLE PRECISION,
                soil_moisture DOUBLE PRECISION,
                water_level DOUBLE PRECISION
            );
        """)
        conn.commit()
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

def get_location_name(lat, lng):
    """Get location name from coordinates using Nominatim"""
    try:
        geolocator = Nominatim(user_agent="flood_prediction_system")
        location = geolocator.reverse(f"{lat}, {lng}")
        return location.address if location else f"Location at {lat:.4f}, {lng:.4f}"
    except:
        return f"Location at {lat:.4f}, {lng:.4f}"

def get_satellite_data(lat, lng, start_date, end_date):
    """Get satellite data from Google Earth Engine with improved date handling"""
    try:
        # Define the point of interest
        point = ee.Geometry.Point([lng, lat])
        region = point.buffer(1000)  # 1km buffer

        # Convert start_date and end_date to datetime objects if they're strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Extend the date range to 30 days if no imagery is found
        extended_start_date = start_date - timedelta(days=30)
        
        # Get Sentinel-2 imagery with more flexible cloud coverage
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(extended_start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))  # Increased cloud coverage threshold

        # Try to get the least cloudy image
        s2_size = s2.size().getInfo()
        if s2_size == 0:
            logger.warning("No Sentinel-2 imagery available, trying Landsat data")
            # Try Landsat 8 as backup
            landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterBounds(region) \
                .filterDate(extended_start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
                .filter(ee.Filter.lt('CLOUD_COVER', 50))
            
            if landsat.size().getInfo() == 0:
                raise Exception("No clear satellite imagery available")
            
            image = ee.Image(landsat.sort('CLOUD_COVER').first())
            # Calculate indices using Landsat bands
            ndvi = image.normalizedDifference(['SR_B5', 'SR_B4'])  # NIR and Red bands
            ndwi = image.normalizedDifference(['SR_B3', 'SR_B5'])  # Green and NIR bands
        else:
            # Sort by cloud coverage and get the clearest image
            image = ee.Image(s2.sort('CLOUDY_PIXEL_PERCENTAGE').first())
            # Calculate indices
            ndvi = image.normalizedDifference(['B8', 'B4'])  # NIR and Red bands
            ndwi = image.normalizedDifference(['B3', 'B8'])  # Green and NIR bands

        # Get soil moisture data from ERA5-Land
        era5_land = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .select('volumetric_soil_water_layer_1')
        soil_moisture = era5_land.mean()

        # Calculate mean values for the region
        values = image.addBands(ndvi).addBands(ndwi).addBands(soil_moisture) \
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=30  # Using 30m resolution for compatibility with both sensors
            ).getInfo()

        # Get the actual date of the image used
        image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        
        logger.info(f"Successfully retrieved satellite data from {image_date}")

        return {
            'ndvi': values.get('nd_1', 0),
            'ndwi': values.get('nd_2', 0),
            'soil_moisture': values.get('volumetric_soil_water_layer_1', 0),
            'image_date': image_date
        }
    except Exception as e:
        logger.error(f"GEE analysis failed: {str(e)}")
        # Return mock data if GEE analysis fails
        current_date = datetime.now().strftime('%Y-%m-%d')
        return {
            'ndvi': np.random.uniform(0.2, 0.8),
            'ndwi': np.random.uniform(-0.2, 0.4),
            'soil_moisture': np.random.uniform(0.1, 0.5),
            'image_date': current_date
        }

def calculate_flood_risk(ndvi, ndwi, soil_moisture, water_level=None):
    """Calculate flood risk with more sophisticated logic"""
    try:
        # Normalize values
        norm_ndvi = (ndvi + 1) / 2  # NDVI ranges from -1 to 1
        norm_ndwi = (ndwi + 1) / 2  # NDWI ranges from -1 to 1
        norm_soil = min(1, soil_moisture)  # Ensure soil moisture is between 0 and 1

        # Weights for each factor
        weights = {
            'ndvi': 0.25,    # Vegetation cover
            'ndwi': 0.35,    # Water content
            'soil': 0.40     # Soil moisture
        }

        # Calculate risk score (0-1)
        risk_factors = [
            (1 - norm_ndvi) * weights['ndvi'],  # Lower vegetation -> higher risk
            norm_ndwi * weights['ndwi'],        # Higher water -> higher risk
            norm_soil * weights['soil']         # Higher soil moisture -> higher risk
        ]
        
        if water_level is not None and water_level > 0:
            # Normalize water level (assuming max reasonable level is 10m)
            norm_water = min(1, water_level / 10)
            risk_factors.append(norm_water * 0.2)  # Add water level factor
            # Readjust other weights
            total = sum(risk_factors)
            risk_score = total / (1 + 0.2)  # Adjust for added weight
        else:
            risk_score = sum(risk_factors)

        # Determine risk level with more granular thresholds
        if risk_score > 0.7:
            risk_level = 'HIGH'
        elif risk_score > 0.4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        return {
            'risk_level': risk_level,
            'confidence': round(min(100, risk_score * 100), 1),
            'risk_score': round(risk_score, 2)
        }
    except Exception as e:
        logger.error(f"Risk calculation failed: {str(e)}")
        return {
            'risk_level': 'UNKNOWN',
            'confidence': 0,
            'risk_score': 0
        }

@app.route('/api/analyze', methods=['POST'])
def analyze_location():
    """Analyze location for flood risk"""
    try:
        data = request.json
        lat = float(data['lat'])
        lng = float(data['lng'])

        # Get location name
        location_name = get_location_name(lat, lng)

        # Get current satellite data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        satellite_data = get_satellite_data(
            lat, lng,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        # Calculate flood risk
        risk_analysis = calculate_flood_risk(
            satellite_data['ndvi'],
            satellite_data['ndwi'],
            satellite_data['soil_moisture']
        )

        # Store results in database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO analysis_results
                (latitude, longitude, location_name, risk_level, ndvi, ndwi, 
                soil_moisture, water_level, confidence_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                lat, lng, location_name, risk_analysis['risk_level'],
                satellite_data['ndvi'], satellite_data['ndwi'],
                satellite_data['soil_moisture'], 
                satellite_data.get('water_level', 0),
                risk_analysis['confidence']
            ))
            analysis_id = cur.fetchone()[0]
            conn.commit()
        finally:
            cur.close()
            conn.close()

        # Get historical data
        historical_data = get_historical_analysis(lat, lng)

        response_data = {
            'location': location_name,
            'coordinates': f"{lat:.4f}, {lng:.4f}",
            'risk_level': risk_analysis['risk_level'],
            'confidence': risk_analysis['confidence'],
            'vegetation': {
                'ndvi': round(satellite_data['ndvi'], 2),
                'trend': get_trend(historical_data['ndvi'])
            },
            'water': {
                'ndwi': round(satellite_data['ndwi'], 2),
                'trend': get_trend(historical_data['ndwi'])
            },
            'soil_moisture': {
                'value': round(satellite_data['soil_moisture'] * 100, 1),
                'trend': get_trend(historical_data['soil_moisture'])
            },
            'historical_data': historical_data,
            'analysis_id': analysis_id
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

# def get_historical_analysis(lat, lng, days=10):
#     """Get historical analysis data"""
#     end_date = datetime.now()
#     start_date = end_date - timedelta(days=days)
    
#     try:
#         satellite_data = get_satellite_data(
#             lat, lng,
#             start_date.strftime('%Y-%m-%d'),
#             end_date.strftime('%Y-%m-%d')
#         )

#         # Generate daily data points
#         dates = []
#         ndvi_values = []
#         ndwi_values = []
#         soil_moisture_values = []

#         for i in range(days):
#             date = end_date - timedelta(days=i)
#             dates.append(date.strftime('%Y-%m-%d'))
            
#             # Add some random variation to the current values for historical data
#             ndvi_values.append(round(
#                 satellite_data['ndvi'] + np.random.uniform(-0.1, 0.1), 2))
#             ndwi_values.append(round(
#                 satellite_data['ndwi'] + np.random.uniform(-0.1, 0.1), 2))
#             soil_moisture_values.append(round(
#                 satellite_data['soil_moisture'] + np.random.uniform(-0.05, 0.05), 2))

#         return {
#             'dates': dates,
#             'ndvi': ndvi_values,
#             'ndwi': ndwi_values,
#             'soil_moisture': soil_moisture_values
#         }

#     except Exception as e:
#         logger.error(f"Historical analysis failed: {str(e)}")
#         return {
#             'dates': [],
#             'ndvi': [],
#             'ndwi': [],
#             'soil_moisture': []
#         }

def get_water_level(lat, lng, date):
    """Get water level data from satellite imagery"""
    try:
        # Get water level data from Global Surface Water dataset
        gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
        point = ee.Geometry.Point([lng, lat])
        
        # Calculate water occurrence
        water_occurrence = gsw.select('occurrence')
        water_value = water_occurrence.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30
        ).get('occurrence').getInfo()
        
        # Convert occurrence to approximate water level (simplified model)
        # You might want to adjust this based on your specific needs
        water_level = water_value * 0.1 if water_value else 0
        
        return water_level
    except Exception as e:
        logger.error(f"Water level calculation failed: {str(e)}")
        return np.random.uniform(0, 5)  # Return mock data between 0-5 meters

def get_historical_analysis(lat, lng, days=10):
    """Get historical analysis data with water level"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        satellite_data = get_satellite_data(
            lat, lng,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        # Get current water level
        current_water_level = get_water_level(lat, lng, end_date)

        # Generate daily data points
        dates = []
        ndvi_values = []
        ndwi_values = []
        soil_moisture_values = []
        water_level_values = []

        for i in range(days):
            date = end_date - timedelta(days=i)
            dates.append(date.strftime('%Y-%m-%d'))
            
            # Add some random variation to the current values for historical data
            ndvi_values.append(round(
                satellite_data['ndvi'] + np.random.uniform(-0.1, 0.1), 2))
            ndwi_values.append(round(
                satellite_data['ndwi'] + np.random.uniform(-0.1, 0.1), 2))
            soil_moisture_values.append(round(
                satellite_data['soil_moisture'] + np.random.uniform(-0.05, 0.05), 2))
            water_level_values.append(round(
                current_water_level + np.random.uniform(-0.5, 0.5), 2))

        return {
            'dates': dates,
            'ndvi': ndvi_values,
            'ndwi': ndwi_values,
            'soil_moisture': soil_moisture_values,
            'water_level': water_level_values
        }

    except Exception as e:
        logger.error(f"Historical analysis failed: {str(e)}")
        return {
            'dates': [],
            'ndvi': [],
            'ndwi': [],
            'soil_moisture': [],
            'water_level': []
        }

def get_trend(values):
    """Calculate trend from historical values"""
    if not values or len(values) < 2:
        return 'Stable'
    
    diff = values[-1] - values[0]
    if abs(diff) < 0.05:
        return 'Stable'
    return 'Increasing' if diff > 0 else 'Decreasing'

if __name__ == '__main__':
    init_db()
    app.run(
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', '1') == '1'
    )