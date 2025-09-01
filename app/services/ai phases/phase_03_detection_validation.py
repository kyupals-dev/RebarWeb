"""
Phase 3: Detection Validation
- Check total detections > 0
- Validate required classes present (front_horizontal, front_vertical)
- If missing either → "No Rebar Detected"
- If both present → Continue to intersection analysis
"""

from .base_phase import BasePhase, PhaseValidationError

class Phase03DetectionValidation(BasePhase):
    """Phase 3: Validate that required rebar classes are detected"""
    
    def __init__(self):
        super().__init__()
        self.phase_name = "Detection Validation"
        self.required_classes = ['front_horizontal', 'front_vertical']
        self.optional_classes = ['back_horizontal']
    
    def validate_input(self, data):
        """Validate input data for Phase 3"""
        required_keys = ['num_detections', 'structured_detections', 'class_names']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate structured_detections format
        detections = data['structured_detections']
        if not isinstance(detections, list):
            raise ValueError("structured_detections must be a list")
        
        return True
    
    def execute(self, data):
        """Execute Phase 3: Detection Validation"""
        self.log(f"Starting {self.phase_name}...")
        
        # Validate input
        self.validate_input(data)
        
        num_detections = data['num_detections']
        detections = data['structured_detections']
        
        self.log(f"Validating {num_detections} detections...")
        
        # Step 1: Check if we have any detections at all
        if num_detections == 0:
            self.log("❌ No detections found")
            return self._create_failure_output(data, "no_detections", "No rebar structures detected in the image")
        
        # Step 2: Analyze detections by class
        class_counts = {}
        valid_detections = []
        
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Count detections by class
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
            
            # Keep track of valid detections
            if confidence > 0.1:  # Additional confidence filter
                valid_detections.append(detection)
        
        self.log(f"Detection counts by class: {class_counts}")
        
        # Step 3: Check for required classes
        missing_classes = []
        for required_class in self.required_classes:
            if required_class not in class_counts or class_counts[required_class] == 0:
                missing_classes.append(required_class)
        
        if missing_classes:
            missing_str = ", ".join(missing_classes)
            self.log(f"❌ Missing required classes: {missing_str}")
            return self._create_failure_output(
                data, 
                "missing_required_classes", 
                f"Missing required rebar classes: {missing_str}"
            )
        
        # Step 4: Validate detection quality
        front_horizontal_detections = [d for d in valid_detections if d['class_name'] == 'front_horizontal']
        front_vertical_detections = [d for d in valid_detections if d['class_name'] == 'front_vertical']
        back_horizontal_detections = [d for d in valid_detections if d['class_name'] == 'back_horizontal']
        
        # Check minimum detection thresholds
        min_front_horizontal = 1
        min_front_vertical = 1
        
        if len(front_horizontal_detections) < min_front_horizontal:
            self.log(f"❌ Insufficient front_horizontal detections: {len(front_horizontal_detections)} < {min_front_horizontal}")
            return self._create_failure_output(
                data,
                "insufficient_front_horizontal",
                f"Need at least {min_front_horizontal} front_horizontal detection(s), found {len(front_horizontal_detections)}"
            )
        
        if len(front_vertical_detections) < min_front_vertical:
            self.log(f"❌ Insufficient front_vertical detections: {len(front_vertical_detections)} < {min_front_vertical}")
            return self._create_failure_output(
                data,
                "insufficient_front_vertical", 
                f"Need at least {min_front_vertical} front_vertical detection(s), found {len(front_vertical_detections)}"
            )
        
        # Step 5: All validations passed
        self.log("✅ All detection validations passed!")
        self.log(f"   front_horizontal: {len(front_horizontal_detections)}")
        self.log(f"   front_vertical: {len(front_vertical_detections)}")
        self.log(f"   back_horizontal: {len(back_horizontal_detections)}")
        
        # Create successful output
        output_data = data.copy()
        output_data.update({
            'validation_passed': True,
            'validation_error': None,
            'class_counts': class_counts,
            'valid_detections': valid_detections,
            'front_horizontal_detections': front_horizontal_detections,
            'front_vertical_detections': front_vertical_detections,
            'back_horizontal_detections': back_horizontal_detections,
            'detection_quality_summary': {
                'total_detections': num_detections,
                'valid_detections': len(valid_detections),
                'front_horizontal_count': len(front_horizontal_detections),
                'front_vertical_count': len(front_vertical_detections),
                'back_horizontal_count': len(back_horizontal_detections)
            }
        })
        
        self.log(f"✅ {self.phase_name} complete: Ready for intersection analysis")
        return output_data
    
    def _create_failure_output(self, data, error_code, error_message):
        """Create output data for validation failure"""
        output_data = data.copy()
        output_data.update({
            'validation_passed': False,
            'validation_error': error_code,
            'validation_message': error_message,
            'should_stop_processing': True  # Signal to stop the pipeline
        })
        
        self.log(f"❌ Validation failed: {error_message}")
        return output_data
    
    def validate_output(self, data):
        """Validate output data from Phase 3"""
        required_keys = ['validation_passed']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing output key: {key}")
        
        # If validation passed, ensure we have the required detection data
        if data['validation_passed']:
            required_success_keys = [
                'front_horizontal_detections', 
                'front_vertical_detections',
                'class_counts',
                'detection_quality_summary'
            ]
            
            for key in required_success_keys:
                if key not in data:
                    raise ValueError(f"Missing success output key: {key}")
        
        # If validation failed, ensure we have error information
        else:
            required_error_keys = ['validation_error', 'validation_message']
            for key in required_error_keys:
                if key not in data:
                    raise ValueError(f"Missing error output key: {key}")
        
        return True
