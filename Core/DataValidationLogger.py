
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from IDataValid import IDataValid

class DataValidationLogger:
	"""Simple logger for data validation operations."""

	def __init__(self, log_dir: str = "logs"):
		self.log_dir = Path(log_dir)
		self.log_dir.mkdir(exist_ok=True)

		# Create log file with timestamp
		self.log_file = self.log_dir / f"data_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

	def log(self, message: str, level: str = "INFO"):
		"""Write message to log file and console."""
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		log_message = f"[{timestamp}] [{level}] {message}"

		# Print to console
		print(log_message)

		# Write to log file
		with open(self.log_file, "a", encoding="utf-8") as f:
			f.write(log_message + "\n")

	def info(self, message: str):
		self.log(message, "INFO")

	def warning(self, message: str):
		self.log(message, "WARNING")

	def error(self, message: str):
		self.log(message, "ERROR")

	def success(self, message: str):
		self.log(message, "SUCCESS")
