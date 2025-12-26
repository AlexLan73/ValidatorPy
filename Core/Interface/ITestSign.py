"""
ITestSign configuration interface for FFT correlation test parameters.
"""


class ITestSign:
    """
    Configuration interface for FFT correlation test parameters.
    
    Stores all configuration parameters needed for FFT correlation algorithm.
    """

    def __init__(self, fft_size: int, num_shifts: int, num_signals: int,
                 num_output_points: int, scale_factor: float):
        """
        Initialize ITestSign configuration.
        
        Args:
            fft_size: FFT window size (e.g., 32768)
                      Must be power of 2 for optimal FFT performance
            
            num_shifts: Number of shifts for reference signal (e.g., 10)
                       Determines shape of reference FFT output
            
            num_signals: Number of input signals to process (e.g., 5)
                        Determines shape of input FFT output
            
            num_output_points: Number of output correlation points (nkg, e.g., 2000)
                              Usually much smaller than fft_size
            
            scale_factor: Scale factor for signal normalization (e.g., 0.000030518)
                         Converts int32 signals to float32: value * scale_factor
                         Typically: 1.0 / fft_size for proper normalization
        """
        self.fft_size = fft_size
        self.num_shifts = num_shifts
        self.num_signals = num_signals
        self.num_output_points = num_output_points
        self.scale_factor = scale_factor
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (f"ITestSign(fft_size={self.fft_size}, num_shifts={self.num_shifts}, "
                f"num_signals={self.num_signals}, num_output_points={self.num_output_points}, "
                f"scale_factor={self.scale_factor})")
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if valid, False otherwise
        """
        # Check if fft_size is power of 2
        if self.fft_size <= 0 or (self.fft_size & (self.fft_size - 1)) != 0:
            print(f"Error: fft_size must be power of 2, got {self.fft_size}")
            return False
        
        # Check positive values
        if self.num_shifts <= 0:
            print(f"Error: num_shifts must be positive, got {self.num_shifts}")
            return False
        
        if self.num_signals <= 0:
            print(f"Error: num_signals must be positive, got {self.num_signals}")
            return False
        
        if self.num_output_points <= 0:
            print(f"Error: num_output_points must be positive, got {self.num_output_points}")
            return False
        
        if self.num_output_points > self.fft_size:
            print(f"Error: num_output_points ({self.num_output_points}) "
                  f"must be <= fft_size ({self.fft_size})")
            return False
        
        if self.scale_factor <= 0:
            print(f"Error: scale_factor must be positive, got {self.scale_factor}")
            return False
        
        return True
