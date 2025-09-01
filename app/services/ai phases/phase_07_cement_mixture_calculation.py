"""
Phase 7: Cement Mixture Calculation
- Use calculated volume (cm³)
- Convert to cubic meters (÷ 1,000,000)
- Apply concrete volume factor (1.5x for rebar cage)
- Calculate Philippine standard ratio (1:2:3): 1 part cement, 2 parts sand, 3 parts aggregate
- Convert to practical units: Cement bags (÷ 0.035 m³ per bag), Sand volume (m³), Aggregate volume (m³)
"""

from .base_phase import BasePhase

class Phase07CementMixtureCalculation(BasePhase):
    """Phase 7: Calculate cement mixture ratios and quantities"""
    
    def __init__(self):
        super().__init__()
        self.phase_name = "Cement Mixture Calculation"
        
        # Philippine standard concrete mix ratios (by volume)
        self.cement_ratio = 1
        self.sand_ratio = 2
        self.aggregate_ratio = 3
        
        # Material conversion factors
        self.cement_bag_volume = 0.035  # cubic meters per bag (40kg bag)
        self.concrete_volume_factor = 1.5  # Multiply rebar volume to get concrete volume
        
        # Safety factors
        self.waste_factor = 1.10  # 10% waste allowance
        self.mixing_efficiency = 0.95  # 5% loss during mixing
    
    def validate_input(self, data):
        """Validate input data for Phase 7"""
        # Check dimension calculation passed
        if not data.get('dimension_calculation_passed', False):
            raise ValueError("Dimension calculation must pass before cement mixture calculation")
        
        required_keys = ['dimensions']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate dimensions structure
        dimensions = data['dimensions']
        if 'volume_cm3' not in dimensions or 'volume_m3' not in dimensions:
            raise ValueError("Dimensions must include volume_cm3 and volume_m3")
        
        if dimensions['volume_cm3'] <= 0 or dimensions['volume_m3'] <= 0:
            raise ValueError("Volume must be positive")
        
        return True
    
    def execute(self, data):
        """Execute Phase 7: Cement Mixture Calculation"""
        self.log(f"Starting {self.phase_name}...")
        
        # Validate input
        self.validate_input(data)
        
        dimensions = data['dimensions']
        
        # Step 1: Get rebar cage volume
        rebar_volume_m3 = dimensions['volume_m3']
        rebar_volume_cm3 = dimensions['volume_cm3']
        
        self.log(f"Rebar cage volume: {rebar_volume_m3:.6f} m³ ({rebar_volume_cm3:.1f} cm³)")
        
        # Step 2: Calculate total concrete volume needed
        # This accounts for concrete around and between rebar
        concrete_volume_m3 = rebar_volume_m3 * self.concrete_volume_factor
        
        self.log(f"Estimated concrete volume: {concrete_volume_m3:.6f} m³ (factor: {self.concrete_volume_factor}x)")
        
        # Step 3: Apply waste and efficiency factors
        adjusted_volume_m3 = concrete_volume_m3 * self.waste_factor / self.mixing_efficiency
        
        self.log(f"Adjusted volume (waste + efficiency): {adjusted_volume_m3:.6f} m³")
        
        # Step 4: Calculate mix ratios
        total_ratio_parts = self.cement_ratio + self.sand_ratio + self.aggregate_ratio
        
        # Calculate volume for each component
        cement_volume_m3 = adjusted_volume_m3 * (self.cement_ratio / total_ratio_parts)
        sand_volume_m3 = adjusted_volume_m3 * (self.sand_ratio / total_ratio_parts)
        aggregate_volume_m3 = adjusted_volume_m3 * (self.aggregate_ratio / total_ratio_parts)
        
        # Step 5: Convert to practical units
        cement_bags = cement_volume_m3 / self.cement_bag_volume
        
        # Round up cement bags to practical quantities
        cement_bags_rounded = max(1, round(cement_bags * 2) / 2)  # Round to nearest 0.5 bag
        
        # Step 6: Calculate material weights (approximate)
        cement_weight_kg = cement_bags_rounded * 40  # 40kg per bag
        sand_weight_kg = sand_volume_m3 * 1600       # ~1600 kg/m³ for sand
        aggregate_weight_kg = aggregate_volume_m3 * 1500  # ~1500 kg/m³ for aggregate
        
        # Step 7: Create mixture data structure
        cement_mixture = {
            # Basic ratios
            'cement_ratio': self.cement_ratio,
            'sand_ratio': self.sand_ratio, 
            'aggregate_ratio': self.aggregate_ratio,
            'ratio_string': f'{self.cement_ratio} Cement : {self.sand_ratio} Sand : {self.aggregate_ratio} Aggregate',
            
            # Volumes
            'total_concrete_volume_m3': round(concrete_volume_m3, 6),
            'adjusted_volume_m3': round(adjusted_volume_m3, 6),
            'cement_volume_m3': round(cement_volume_m3, 6),
            'sand_volume_m3': round(sand_volume_m3, 6),
            'aggregate_volume_m3': round(aggregate_volume_m3, 6),
            
            # Practical quantities
            'cement_bags': round(cement_bags, 2),
            'cement_bags_rounded': cement_bags_rounded,
            'sand_volume_m3_practical': round(sand_volume_m3, 4),
            'aggregate_volume_m3_practical': round(aggregate_volume_m3, 4),
            
            # Weights (approximate)
            'cement_weight_kg': round(cement_weight_kg, 1),
            'sand_weight_kg': round(sand_weight_kg, 1),
            'aggregate_weight_kg': round(aggregate_weight_kg, 1),
            'total_weight_kg': round(cement_weight_kg + sand_weight_kg + aggregate_weight_kg, 1),
            
            # Calculation metadata
            'calculation_method': 'philippine_standard_mix',
            'concrete_volume_factor': self.concrete_volume_factor,
            'waste_factor': self.waste_factor,
            'mixing_efficiency': self.mixing_efficiency,
            'cement_bag_size_kg': 40
        }
        
        # Step 8: Create summary for display
        mixture_summary = self._create_mixture_summary(cement_mixture)
        
        # Create output data
        output_data = data.copy()
        output_data.update({
            'cement_mixture_calculation_passed': True,
            'cement_mixture': cement_mixture,
            'mixture_summary': mixture_summary
        })
        
        self.log(f"✅ {self.phase_name} complete:")
        self.log(f"   Mix ratio: {cement_mixture['ratio_string']}")
        self.log(f"   Cement: {cement_mixture['cement_bags_rounded']} bags ({cement_mixture['cement_weight_kg']} kg)")
        self.log(f"   Sand: {cement_mixture['sand_volume_m3_practical']} m³ ({cement_mixture['sand_weight_kg']} kg)")
        self.log(f"   Aggregate: {cement_mixture['aggregate_volume_m3_practical']} m³ ({cement_mixture['aggregate_weight_kg']} kg)")
        self.log(f"   Total concrete: {cement_mixture['total_concrete_volume_m3']} m³")
        
        return output_data
    
    def _create_mixture_summary(self, mixture):
        """Create human-readable mixture summary"""
        return {
            'ratio': mixture['ratio_string'],
            'materials': {
                'cement': f"{mixture['cement_bags_rounded']} bags ({mixture['cement_weight_kg']} kg)",
                'sand': f"{mixture['sand_volume_m3_practical']} m³ ({mixture['sand_weight_kg']} kg)",
                'aggregate': f"{mixture['aggregate_volume_m3_practical']} m³ ({mixture['aggregate_weight_kg']} kg)"
            },
            'totals': {
                'concrete_volume': f"{mixture['total_concrete_volume_m3']} m³",
                'total_weight': f"{mixture['total_weight_kg']} kg",
                'estimated_cost_factors': {
                    'cement_bags': mixture['cement_bags_rounded'],
                    'sand_m3': mixture['sand_volume_m3_practical'],
                    'aggregate_m3': mixture['aggregate_volume_m3_practical']
                }
            },
            'notes': [
                f"Includes {int((self.waste_factor - 1) * 100)}% waste allowance",
                f"Based on {self.concrete_volume_factor}x volume factor for rebar cage",
                "Weights are approximate - actual weights may vary by material source",
                "Philippine standard mix suitable for residential construction"
            ]
        }
    
    def validate_output(self, data):
        """Validate output data from Phase 7"""
        required_keys = ['cement_mixture_calculation_passed', 'cement_mixture']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing output key: {key}")
        
        # Validate cement mixture structure
        mixture = data['cement_mixture']
        required_mixture_keys = [
            'ratio_string', 'cement_bags', 'sand_volume_m3', 'aggregate_volume_m3',
            'total_concrete_volume_m3', 'cement_weight_kg', 'sand_weight_kg', 'aggregate_weight_kg'
        ]
        
        for key in required_mixture_keys:
            if key not in mixture:
                raise ValueError(f"Missing cement mixture key: {key}")
        
        # Validate values are positive
        positive_values = [
            'cement_bags', 'sand_volume_m3', 'aggregate_volume_m3', 
            'cement_weight_kg', 'sand_weight_kg', 'aggregate_weight_kg'
        ]
        
        for key in positive_values:
            if mixture[key] <= 0:
                raise ValueError(f"Cement mixture {key} must be positive, got {mixture[key]}")
        
        return True
