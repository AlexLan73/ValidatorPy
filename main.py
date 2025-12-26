
from DataLoader import DataLoader


# Example usage
if __name__ == "__main__":
	# Initialize loader
	loader = DataLoader(
		validation_data_dir="Data/Validation",
		source_json_dir="/home/alex/C++/Correlator/Report/Validation"
	)

	# Load data (will convert from JSON if needed)
	data = loader.load_data("2025-12-25_11-06-12")

	if data:
		print(f"Successfully loaded data")
		print(f"Reference signal shape: {data.dstep0[0].shape if data.dstep0 else 'None'}")
		print(f"ITestSign config: fft_size={data.its.fft_size}, scale_factor={data.its.scale_factor}")
	else:
		print("Failed to load data")
