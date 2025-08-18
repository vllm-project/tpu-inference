#!/usr/bin/env python3
"""
Simple script that starts two processes on TPU, each using 4 chips.
Each process prints out its local devices.

Assumes 8 total TPU chips available.
"""

import multiprocessing
import os
import jax

# Environment variable constants
TPU_CHIPS_PER_PROCESS_BOUNDS = "TPU_CHIPS_PER_PROCESS_BOUNDS"
TPU_PROCESS_BOUNDS = "TPU_PROCESS_BOUNDS"
TPU_VISIBLE_CHIPS = "TPU_VISIBLE_CHIPS"


def process_worker(process_id: int, visible_chips: str, process_name: str):
    """Worker function that runs in each process."""
    
    def log(message: str):
        print(f"[{process_name}] {message}")
    
    log(f"Starting process {process_id}")
    
    # Set TPU environment variables to limit visible chips
    os.environ[TPU_CHIPS_PER_PROCESS_BOUNDS] = "1,4,1"
    os.environ[TPU_PROCESS_BOUNDS] = "1,1,1" 
    os.environ[TPU_VISIBLE_CHIPS] = visible_chips
    
    log(f"Set TPU_VISIBLE_CHIPS to: {visible_chips}")
    
    # Print device information
    try:
        local_devices = jax.local_devices()
        global_devices = jax.devices()
        
        log(f"Local device count: {jax.local_device_count()}")
        log(f"Global device count: {jax.device_count()}")
        
        log("Local devices:")
        for i, device in enumerate(local_devices):
            log(f"  Device {i}: {device}")
            
        log("All global devices:")
        for i, device in enumerate(global_devices):
            log(f"  Global device {i}: {device}")
            
    except Exception as e:
        log(f"Error initializing JAX devices: {e}")
    
    log(f"Process {process_id} completed")


def main():
    """Main function that spawns two processes."""
    
    print("Starting JAX multi-process TPU script")
    print("Spawning 2 processes, each with 4 TPU chips")
    
    # Process 1: Use chips 0,1,2,3
    process1 = multiprocessing.get_context("fork").Process(
        target=process_worker,
        args=(1, "0,1,2,3", "Process-1")
    )
    
    # Process 2: Use chips 4,5,6,7  
    process2 = multiprocessing.get_context("fork").Process(
        target=process_worker,
        args=(2, "4,5,6,7", "Process-2")
    )
    
    # Start both processes
    print("Starting Process-1 with chips 0,1,2,3")
    process1.start()
    
    print("Starting Process-2 with chips 4,5,6,7")
    process2.start()
    
    # Wait for both processes to complete
    process1.join()
    process2.join()
    
    print("Both processes completed")


if __name__ == "__main__":
    main()