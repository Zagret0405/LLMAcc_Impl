import json
from pydantic import BaseModel, Field

class HardwareSpec(BaseModel):
    """
    Schema for validating hardware specs loaded from JSON.
    """
    name: str = Field(..., description="Device name")
    total_macs: int = Field(..., description="Number of MAC units")
    frequency_hz: float = Field(..., description="Operating frequency (Hz)")
    on_chip_mem_bytes: int = Field(..., description="On-chip memory size in bytes")
    num_bram_blocks: int = Field(..., description="Number of BRAM blocks")
    bitwidth_bram: int = Field(..., description="Bitwidth of each BRAM block")
    on_chip_mem_ports_per_block: int = Field(..., description="Ports per BRAM block")
    on_chip_mem_bw_per_port_bits_per_cycle: int = Field(..., description="Bandwidth per port (bits/cycle)")
    off_chip_mem_bw_bytes_per_sec: float = Field(..., description="Off-chip memory bandwidth (bytes/sec)")

class HardwareDevice:
    """
    Stores the physical specifications of the target hardware device.
    """
    def __init__(self, name, total_macs, frequency_hz, 
                 on_chip_mem_bytes, num_bram_blocks, bitwidth_bram,
                 on_chip_mem_ports_per_block, 
                 on_chip_mem_bw_per_port_bits_per_cycle, 
                 off_chip_mem_bw_bytes_per_sec):
        self.name = name
        self.total_macs = total_macs
        self.frequency_hz = frequency_hz
        self.on_chip_mem_bytes = on_chip_mem_bytes
        self.num_bram_blocks = num_bram_blocks
        self.bitwidth_bram = bitwidth_bram
        self.on_chip_mem_ports_per_block = on_chip_mem_ports_per_block
        self.on_chip_mem_bw_per_port_bits_per_cycle = on_chip_mem_bw_per_port_bits_per_cycle
        self.off_chip_mem_bw_bytes_per_sec = off_chip_mem_bw_bytes_per_sec

    @classmethod
    def from_json(cls, file_path: str, device_name: str):
        """
        Factory method to create a HardwareDevice instance from a JSON file.

        Args:
            file_path (str): The path to the JSON configuration file.
            device_name (str): The name of the device to load from the JSON file.

        Returns:
            A HardwareDevice instance.
        """
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        device_data = config_data.get(device_name)
        if not device_data:
            raise ValueError(f"Device '{device_name}' not found in {file_path}")

        # Validate data using the pydantic schema
        validated_data = HardwareSpec(**device_data)
        
        # Create instance from validated data
        return cls(
            name=validated_data.name,
            total_macs=validated_data.total_macs,
            frequency_hz=validated_data.frequency_hz,
            on_chip_mem_bytes=validated_data.on_chip_mem_bytes,
            num_bram_blocks=validated_data.num_bram_blocks,
            bitwidth_bram=validated_data.bitwidth_bram,
            on_chip_mem_ports_per_block=validated_data.on_chip_mem_ports_per_block,
            on_chip_mem_bw_per_port_bits_per_cycle=validated_data.on_chip_mem_bw_per_port_bits_per_cycle,
            off_chip_mem_bw_bytes_per_sec=validated_data.off_chip_mem_bw_bytes_per_sec
        )

    def __repr__(self):
        return (f"HardwareDevice(name={self.name}, total_macs={self.total_macs}, "
                f"on_chip_mem_mb={self.on_chip_mem_bytes / 1e6:.2f})")