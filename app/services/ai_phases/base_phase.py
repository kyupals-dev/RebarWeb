"""
Base Phase Class
Provides common interface and utilities for all AI processing phases
"""

from abc import ABC, abstractmethod
from datetime import datetime

class BasePhase(ABC):
    """Abstract base class for all AI processing phases"""
    
    def __init__(self):
        self.phase_name = "Unnamed Phase"
        self.start_time = None
        self.end_time = None
        self.execution_time = None
    
    @abstractmethod
    def execute(self, data):
        """
        Execute the phase processing
        
        Args:
            data (dict): Input data from previous phase or initial input
            
        Returns:
            dict: Output data for next phase
        """
        pass
    
    @abstractmethod
    def validate_input(self, data):
        """
        Validate input data before processing
        
        Args:
            data (dict): Input data to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If input is invalid
        """
        pass
    
    def validate_output(self, data):
        """
        Validate output data after processing (optional override)
        
        Args:
            data (dict): Output data to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If output is invalid
        """
        return True
    
    def log(self, message):
        """Log a message with phase name and timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[{timestamp}] {self.phase_name}: {message}")
    
    def start_timing(self):
        """Start timing the phase execution"""
        self.start_time = datetime.now()
    
    def end_timing(self):
        """End timing and calculate execution time"""
        self.end_time = datetime.now()
        if self.start_time:
            self.execution_time = (self.end_time - self.start_time).total_seconds()
    
    def get_timing_info(self):
        """Get timing information for this phase"""
        return {
            'phase_name': self.phase_name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'execution_time_seconds': self.execution_time
        }
    
    def execute_with_timing(self, data):
        """Execute phase with automatic timing"""
        self.start_timing()
        try:
            result = self.execute(data)
            self.validate_output(result)
            return result
        finally:
            self.end_timing()
            self.log(f"Completed in {self.execution_time:.3f}s")

class PhaseError(Exception):
    """Custom exception for phase processing errors"""
    
    def __init__(self, phase_name, message, original_error=None):
        self.phase_name = phase_name
        self.original_error = original_error
        super().__init__(f"{phase_name}: {message}")

class PhaseValidationError(PhaseError):
    """Exception for phase input/output validation errors"""
    pass
