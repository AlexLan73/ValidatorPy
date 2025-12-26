"""
IDataValid and ITestSign interfaces for validation data.
"""

import ITestSign as its


class IDataValid:
    """
    Data validation interface for FFT correlation results.
    
    Stores 4 steps of FFT correlation processing:
    - dstep0: Reference and input signals (M-sequences)
    - dstep1: Reference FFT
    - dstep2: Input FFT
    - dstep3: Correlation results (peaks)
    """
    
    def __init__(self, fft_size: int, num_shifts: int, num_signals: int, 
                 num_output_points: int, scale_factor: float):
        """
        Initialize IDataValid.
        
        Args:
            fft_size: FFT window size (e.g., 32768)
            num_shifts: Number of shifts for reference signal (e.g., 10)
            num_signals: Number of input signals (e.g., 5)
            num_output_points: Number of output points (nkg, e.g., 2000)
            scale_factor: Scale factor for signal normalization (e.g., 0.000030518)
        """
        self.dstep0 = None  # Tuple of (reference_signal, input_signals)
        self.dstep1 = None  # Reference FFT array
        self.dstep2 = None  # Input FFT array
        self.dstep3 = None  # Correlation peaks array
        
        # Store configuration in ITestSign object
        self.its = its.ITestSign(
            fft_size=fft_size,
            num_shifts=num_shifts,
            num_signals=num_signals,
            num_output_points=num_output_points,
            scale_factor=scale_factor
        )
    
    def set_step(self, d_step0, d_step1, d_step2, d_step3):
        """
        Set all step data at once.
        
        Args:
            d_step0: Reference and input signals (tuple or array)
            d_step1: Reference FFT array
            d_step2: Input FFT array
            d_step3: Correlation peaks array
        """
        self.dstep0 = d_step0
        self.dstep1 = d_step1
        self.dstep2 = d_step2
        self.dstep3 = d_step3
    
    def get_config(self) -> dict:
        """
        Get configuration parameters.
        
        Returns:
            Dictionary with configuration
        """
        return {
            "fft_size": self.its.fft_size,
            "num_shifts": self.its.num_shifts,
            "num_signals": self.its.num_signals,
            "num_output_points": self.its.num_output_points,
            "scale_factor": self.its.scale_factor
        }
