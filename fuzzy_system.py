"""
Fuzzy Logic System for Medical Severity Classification
Combines CNN feature vectors with domain knowledge through fuzzy rules
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def build_fuzzy_system():
    """
    Build complete fuzzy logic inference system for severity classification.
    
    System uses 3 fuzzy INPUT variables (extracted from CNN feature vectors):
    1. tumor_intensity: How bright/strong the tumor appears (0-1)
    2. texture_irregularity: How rough/jagged the tumor boundary is (0-1)
    3. boundary_clarity: How sharp/defined the tumor edge is (0-1)
    
    And 1 OUTPUT variable:
    - severity: Predicted severity score (0-3)
      0.0-1.0 = Normal/Mild
      1.0-2.0 = Mild/Moderate
      2.0-3.0 = Moderate/Severe
    
    8 Fuzzy Rules based on medical imaging knowledge:
    Rule 1: low_intensity AND smooth_texture → normal
    Rule 2: low_intensity AND rough_texture → mild
    Rule 3: medium_intensity AND rough_texture → mild
    Rule 4: medium_intensity AND very_rough → moderate
    Rule 5: high_intensity AND very_rough → severe
    Rule 6: high_intensity AND blurry_boundary → severe
    Rule 7: low_intensity AND clear_boundary → normal
    Rule 8: medium_intensity AND blurry_boundary → moderate
    
    Membership Functions:
    - Triangular fuzzy sets covering the input/output ranges
    - Overlapping sets allow smooth transitions between categories
    
    Returns:
        ControlSystemSimulation object ready for inference
    """
    
    # ===== INPUT VARIABLE 1: Tumor Intensity =====
    tumor_intensity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 
                                      'tumor_intensity')
    tumor_intensity['low'] = fuzz.trimf(tumor_intensity.universe, 
                                         [0.0, 0.0, 0.4])
    tumor_intensity['medium'] = fuzz.trimf(tumor_intensity.universe, 
                                            [0.2, 0.5, 0.8])
    tumor_intensity['high'] = fuzz.trimf(tumor_intensity.universe, 
                                          [0.6, 1.0, 1.0])
    
    # ===== INPUT VARIABLE 2: Texture Irregularity =====
    texture_irregularity = ctrl.Antecedent(np.arange(0, 1.01, 0.01),
                                           'texture_irregularity')
    texture_irregularity['smooth'] = fuzz.trimf(texture_irregularity.universe, 
                                                 [0.0, 0.0, 0.4])
    texture_irregularity['slightly_rough'] = fuzz.trimf(texture_irregularity.universe, 
                                                         [0.2, 0.5, 0.8])
    texture_irregularity['very_rough'] = fuzz.trimf(texture_irregularity.universe, 
                                                     [0.6, 1.0, 1.0])
    
    # ===== INPUT VARIABLE 3: Boundary Clarity =====
    boundary_clarity = ctrl.Antecedent(np.arange(0, 1.01, 0.01),
                                       'boundary_clarity')
    boundary_clarity['blurry'] = fuzz.trimf(boundary_clarity.universe,
                                             [0.0, 0.0, 0.4])
    boundary_clarity['clear'] = fuzz.trimf(boundary_clarity.universe,
                                            [0.6, 1.0, 1.0])
    
    # ===== OUTPUT VARIABLE: Severity Score =====
    severity = ctrl.Consequent(np.arange(0, 3.01, 0.01), 'severity')
    severity['normal'] = fuzz.trimf(severity.universe, [0.0, 0.0, 0.8])
    severity['mild'] = fuzz.trimf(severity.universe, [0.6, 1.0, 1.4])
    severity['moderate'] = fuzz.trimf(severity.universe, [1.2, 1.8, 2.2])
    severity['severe'] = fuzz.trimf(severity.universe, [2.0, 3.0, 3.0])
    
    # ===== FUZZY RULES =====
    # Simplified rules to avoid conflicts
    rule1 = ctrl.Rule(tumor_intensity['low'], severity['normal'])
    rule2 = ctrl.Rule(tumor_intensity['medium'], severity['mild'])
    rule3 = ctrl.Rule(tumor_intensity['high'], severity['severe'])
    rule4 = ctrl.Rule(texture_irregularity['very_rough'], severity['moderate'])
    rule5 = ctrl.Rule(boundary_clarity['blurry'], severity['moderate'])
    
    # Create control system from simplified rules
    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    
    # Create simulation for inference
    fuzzy_sim = ctrl.ControlSystemSimulation(system)
    
    return fuzzy_sim


def extract_fuzzy_inputs(feature_vector):
    """
    Extract 3 fuzzy input values from 256-dimensional CNN feature vector.
    
    The 256-dim feature vector captures high-level image characteristics.
    We divide it into 3 regions and extract domain-relevant features:
    
    1. First 85 dims (0:85) → tumor_intensity
       Average these to get overall intensity/brightness
       
    2. Middle 85 dims (85:170) → texture_irregularity
       Standard deviation indicates texture roughness
       Multiply by 3 to scale to 0-1 range
       
    3. Last 86 dims (170:256) → boundary_clarity
       Average to get boundary definition
    
    All values normalized to 0-1 range and clipped to 0.01-0.99
    (Avoid exactly 0 or 1 which causes fuzzy system errors)
    
    Args:
        feature_vector: 1D numpy array of 256 values
                       (from VGG16 Dense(256) layer)
                       
    Returns:
        Dict with 3 fuzzy inputs, all in range [0.01, 0.99]
    """
    # Normalize feature vector to 0-1 range
    v = feature_vector.copy()
    v_min = v.min()
    v_max = v.max()
    
    # Avoid division by zero
    if v_max - v_min < 1e-8:
        v = np.ones_like(v) * 0.5
    else:
        v = (v - v_min) / (v_max - v_min + 1e-8)
    
    # Extract 3 fuzzy input values from different regions of feature vector
    
    # Tumor intensity: average of first 85 dimensions
    tumor_intensity = float(np.mean(v[0:85]))
    
    # Texture irregularity: standard deviation of middle 85 dimensions, scaled by 3
    texture_irregularity = float(np.std(v[85:170]) * 3)
    
    # Boundary clarity: average of last 86 dimensions
    boundary_clarity = float(np.mean(v[170:256]))
    
    # Clip all values to safe range [0.01, 0.99]
    # Avoid exactly 0 or 1 to prevent fuzzy system numerical issues
    tumor_intensity = np.clip(tumor_intensity, 0.01, 0.99)
    texture_irregularity = np.clip(texture_irregularity, 0.01, 0.99)
    boundary_clarity = np.clip(boundary_clarity, 0.01, 0.99)
    
    return {
        'tumor_intensity': tumor_intensity,
        'texture_irregularity': texture_irregularity,
        'boundary_clarity': boundary_clarity
    }


def fuzzy_classify(feature_vector, fuzzy_sim):
    """
    Classify using the fuzzy inference system and feature-derived inputs.

    The feature vector is converted to interpretable fuzzy inputs, then
    the fuzzy control system is evaluated to produce a continuous severity score.
    The score is discretized into one of the four output classes.
    """
    try:
        # Extract fuzzy inputs from the feature vector
        inputs = extract_fuzzy_inputs(feature_vector)

        # Feed inputs into the fuzzy system
        fuzzy_sim.input['tumor_intensity'] = inputs['tumor_intensity']
        fuzzy_sim.input['texture_irregularity'] = inputs['texture_irregularity']
        fuzzy_sim.input['boundary_clarity'] = inputs['boundary_clarity']
        fuzzy_sim.compute()

        severity_score = float(fuzzy_sim.output['severity'])
        severity_score = np.clip(severity_score, 0.0, 3.0)

        if severity_score < 0.8:
            predicted_class = 0
        elif severity_score < 1.4:
            predicted_class = 1
        elif severity_score < 2.1:
            predicted_class = 2
        else:
            predicted_class = 3

        return (predicted_class, severity_score, inputs, False)

    except Exception:
        fallback_class = int(np.clip(np.round(np.mean(feature_vector) * 3.0), 0, 3))
        return (fallback_class, float(fallback_class), {
            'tumor_intensity': 0.5,
            'texture_irregularity': 0.5,
            'boundary_clarity': 0.5
        }, True)

