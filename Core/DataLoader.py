"""
Static class for loading/saving validation data with JSON-to-pickle conversion.
Handles multi-step FFT correlation data (Step0, Step1, Step2, Step3).
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from IDataValid import IDataValid
from DataValidationLogger import DataValidationLogger


class DataLoader:
    """Static class for loading and saving validation data."""

    # Current version for pickle compatibility checking
    PICKLE_VERSION = 1

    def __init__(self, validation_data_dir: str = "Data/Validation",
                 source_json_dir: str = "/home/alex/C++/Correlator/Report/Validation",
                 log_dir: str = "logs"):
        """
        Initialize DataLoader with directories.

        Args:
            validation_data_dir: Directory for converted pickle files (relative to project root)
            source_json_dir: Directory with source JSON files (absolute path)
            log_dir: Directory for log files
        """
        self.validation_data_dir = Path(validation_data_dir)
        self.source_json_dir = Path(source_json_dir)
        self.logger = DataValidationLogger(log_dir)

        # Create validation directory if it doesn't exist
        self.validation_data_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"DataLoader initialized")
        self.logger.info(
            f"Validation data dir: {
                self.validation_data_dir.absolute()}")
        self.logger.info(f"Source JSON dir: {self.source_json_dir.absolute()}")

    def load_data(self, data_name: str) -> Optional[IDataValid]:
        """
        Load validation data by name.

        First tries to load from pickle cache, if not found or invalid,
        converts from JSON and saves to pickle.

        Args:
            data_name: Name of the dataset (without extension)

        Returns:
            IDataValid object or None if loading failed
        """
        self.logger.info(f"Loading data: {data_name}")

        # Try to load from pickle cache
        pickle_path = self.validation_data_dir / f"{data_name}.pkl"

        if pickle_path.exists():
            try:
                data = self._load_from_pickle(pickle_path)
                if data is not None:
                    self.logger.success(
                        f"Loaded from pickle cache: {data_name}")
                    return data
            except Exception as e:
                self.logger.warning(f"Failed to load pickle: {str(e)}")

        # Try to load from JSON and convert
        self.logger.info(
            f"Pickle not found or invalid, trying JSON source: {data_name}")
        data = self._load_from_json_and_convert(data_name)

        if data is not None:
            # Save to pickle
            try:
                self._save_to_pickle(data, pickle_path)
                self.logger.success(f"Saved to pickle: {data_name}")
            except Exception as e:
                self.logger.error(f"Failed to save pickle: {str(e)}")
                return None

            return data

        self.logger.error(f"Failed to load data: {data_name}")
        return None

    def _load_from_pickle(self, pickle_path: Path) -> Optional[IDataValid]:
        """
        Load and validate pickle file.

        Args:
            pickle_path: Path to pickle file

        Returns:
            IDataValid object or None if validation failed
        """
        with open(pickle_path, "rb") as f:
            data_dict = pickle.load(f)

        # Check version compatibility
        version = data_dict.get("version")
        if version != self.PICKLE_VERSION:
            self.logger.warning(
                f"Pickle version mismatch: expected {
                    self.PICKLE_VERSION}, got {version}"
            )
            return None

        # Reconstruct IDataValid object
        data_obj = data_dict["data"]

        return data_obj

    def _save_to_pickle(self, data: IDataValid, pickle_path: Path) -> None:
        """
        Save IDataValid object to pickle with version info.

        Args:
            data: IDataValid object to save
            pickle_path: Path where to save pickle file
        """
        data_dict = {
            "version": self.PICKLE_VERSION,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

        with open(pickle_path, "wb") as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_from_json_and_convert(
            self, data_name: str) -> Optional[IDataValid]:
        """
        Load data from 4 JSON files (Step0-Step3) and convert to IDataValid.

        Args:
            data_name: Name of the dataset directory

        Returns:
            IDataValid object or None if conversion failed
        """
        # Construct path to data directory
        data_dir = self.source_json_dir / data_name

        if not data_dir.exists():
            self.logger.error(f"Data directory not found: {data_dir}")
            return None

        try:
            # Load all 4 step files
            step_files = [
                "Step0.json",
                "Step1.json",
                "Step2.json",
                "Step3.json"]
            steps_data = []

            for step_file in step_files:
                step_path = data_dir / step_file

                if not step_path.exists():
                    self.logger.error(f"Missing step file: {step_path}")
                    return None

                with open(step_path, "r") as f:
                    step_data = json.load(f)
                    steps_data.append(step_data)
                    self.logger.info(f"Loaded: {step_file}")

            # Extract configuration from Step0
            config = steps_data[0].get("configuration", {})
            fft_size = config.get("fft_size")
            num_shifts = config.get("num_shifts")
            num_signals = config.get("num_signals")
            num_output_points = config.get("num_output_points")
            scale_factor = config.get("scale_factor")

            # Validate configuration
            if not all([fft_size, num_shifts, num_signals,
                       num_output_points, scale_factor is not None]):
                self.logger.error(
                    f"Invalid or incomplete configuration in Step0.json")
                return None

            self.logger.info(
                f"Configuration: fft_size={fft_size}, num_shifts={num_shifts}, "
                f"num_signals={num_signals}, num_output_points={num_output_points}, "
                f"scale_factor={scale_factor}"
            )

            # Create IDataValid object
            data_valid = IDataValid(
                fft_size=fft_size,
                num_shifts=num_shifts,
                num_signals=num_signals,
                num_output_points=num_output_points,
                scale_factor=scale_factor
            )

            # Parse and convert step data
            dstep0 = self._parse_step0(steps_data[0])
            dstep1 = self._parse_step1(steps_data[1])
            dstep2 = self._parse_step2(steps_data[2])
            dstep3 = self._parse_step3(steps_data[3], num_output_points)

            # Set data in IDataValid
            data_valid.set_step(dstep0, dstep1, dstep2, dstep3)

            self.logger.success(
                f"Successfully converted {data_name} from JSON")
            return data_valid

        except Exception as e:
            self.logger.error(f"Failed to convert JSON data: {str(e)}")
            return None

    @staticmethod
    def _parse_step0(step0_data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse Step0 (M-sequence generation).

        Returns:
              Tuple of (reference_signal, input_signals) as numpy arrays
        """
        reference_signal = np.array(
            step0_data.get(
                "reference_signal",
                []),
            dtype=np.int32)
        input_signals = np.array(
            step0_data.get(
                "input_signals",
                []),
            dtype=np.int32)

        return reference_signal, input_signals

    @staticmethod
    def _parse_step1(step1_data: dict) -> np.ndarray:
        """
        Parse Step1 (Reference FFT).

        Returns:
            Reference FFT as complex64 array (numshifts, fftsize)
        """
        data = step1_data.get("data", {})
        reference_fft_list = data.get("reference_fft", [])

        # Convert list of {real, imag, shift, pos} to complex64 array
        reference_fft = np.zeros(
            (data.get("num_shifts", 0), data.get("fft_size", 0)),
            dtype=np.complex64
        )

        for item in reference_fft_list:
            shift = item.get("shift", 0)
            pos = item.get("pos", 0)
            real = item.get("real", 0.0)
            imag = item.get("imag", 0.0)
            reference_fft[shift, pos] = complex(real, imag)

        return reference_fft

    @staticmethod
    def _parse_step2(step2_data: dict) -> np.ndarray:
        """
        Parse Step2 (Input FFT).

        Returns:
            Input FFT as complex64 array (numsignals, fftsize)
        """
        data = step2_data.get("data", {})
        input_fft_list = data.get('input_fft', [])

        # Convert list of {real, imag, signal, pos} to complex64 array
        input_fft = np.zeros(
            (data.get("numsignals", 0), data.get("fftsize", 0)),
            dtype=np.complex64
        )

        for item in input_fft_list:
            signal = item.get("signal", 0)
            pos = item.get("pos", 0)
            real = item.get("real", 0.0)
            imag = item.get("imag", 0.0)
            input_fft[signal, pos] = complex(real, imag)

        return input_fft

    @staticmethod
    def _parse_step3(step3_data: dict, num_output_points: int) -> np.ndarray:
        """
        Parse Step3 (Correlation results).

        Args:
            step3_data: Step3 JSON data
            num_output_points: Expected number of output points (numoutputpoints)

        Returns:
            Peaks as float32 array (numsignals, numshifts, num_output_points)
        """
        data = step3_data.get("data", {})
        peaks_list = data.get("peaks", [])

        # Convert list of peaks to float32 array
        peaks = np.zeros(
            (data.get("numsignals", 0), data.get(
                "numshifts", 0), num_output_points),
            dtype=np.float32
        )

        for item in peaks_list:
            signal = item.get("signal", 0)
            shift = item.get("shift", 0)
            nkg_values = item.get("nkg", [])

            for idx, value in enumerate(nkg_values[:num_output_points]):
                peaks[signal, shift, idx] = float(value)

        return peaks
