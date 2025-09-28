import subprocess
import csv
import time
import re
from datetime import datetime

# --- Configuration ---
ADB_PATH = "adb"  # Assumes 'adb' is in your system's PATH
OUTPUT_FILE = "qidk_performance_log.csv"
SAMPLE_INTERVAL_SECONDS = 5  # Time between samples

# --- ADB Command Wrappers ---

def run_adb_shell_command(command):
    """Executes an ADB shell command and returns the output string."""
    try:
        result = subprocess.run(
            [ADB_PATH, "shell", command],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"ADB Error executing '{command}': {e.stderr.strip()}")
        return ""
    except FileNotFoundError:
        print("Error: ADB not found. Ensure 'adb' is in your system PATH.")
        return ""

# --- Metric Functions ---

def get_cpu_usage(prev_cpu_data):
    """Calculates total CPU usage (%) by parsing /proc/stat."""
    try:
        raw_output = run_adb_shell_command("cat /proc/stat")
        if not raw_output:
            return None, prev_cpu_data

        # Find the main 'cpu' line
        cpu_line = [line for line in raw_output.splitlines() if line.startswith("cpu ")][0]
        # Values: user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice
        current_data = list(map(int, cpu_line.split()[1:8]))
        
        # Calculate total usage if previous data exists
        if prev_cpu_data:
            prev_idle = prev_cpu_data[3]
            current_idle = current_data[3]
            
            prev_total = sum(prev_cpu_data)
            current_total = sum(current_data)
            
            total_diff = current_total - prev_total
            idle_diff = current_idle - prev_idle
            
            # Prevent division by zero if total_diff is 0
            if total_diff > 0:
                cpu_percent = 100.0 * (total_diff - idle_diff) / total_diff
                return round(cpu_percent, 2), current_data
        
        return None, current_data # Return current data for next iteration

    except Exception as e:
        print(f"Error calculating CPU usage: {e}")
        return None, prev_cpu_data


def get_ram_usage():
    """Calculates used RAM percentage and MB by parsing /proc/meminfo."""
    try:
        raw_output = run_adb_shell_command("cat /proc/meminfo")
        if not raw_output:
            return None, None

        mem_info = {}
        for line in raw_output.splitlines():
            name, value, unit = line.split()
            mem_info[name[:-1]] = int(value) * 1024 # Convert KB to Bytes

        total_mem_b = mem_info.get("MemTotal", 0)
        free_mem_b = mem_info.get("MemAvailable", 0) # Use MemAvailable for accuracy
        
        if total_mem_b > 0:
            used_mem_b = total_mem_b - free_mem_b
            used_mem_mb = round(used_mem_b / (1024 * 1024), 2)
            used_percent = round((used_mem_b / total_mem_b) * 100, 2)
            return used_percent, used_mem_mb
        
        return None, None

    except Exception as e:
        print(f"Error calculating RAM usage: {e}")
        return None, None

def get_temperature():
    """
    Reads the maximum temperature from thermal zones.
    Assumes centi-degrees (x100) or deci-degrees (x10) based on observed data.
    The most common large multipliers are 1000 or 100. Since 90-95 is too high, 
    we test against a 100x multiplier first to get a reasonable value (e.g., 30-45C).
    """
    try:
        # Check common thermal zones and find the maximum temperature
        thermal_zones = run_adb_shell_command("ls /sys/class/thermal/ | grep thermal_zone")
        max_temp_c = 0.0

        for zone in thermal_zones.splitlines():
            temp_str = run_adb_shell_command(f"cat /sys/class/thermal/{zone}/temp")
            
            if temp_str and temp_str.isdigit():
                temp_raw = int(temp_str)
                temp_c = 0.0

                # **CRITICAL CHANGE HERE**
                # Assume raw value is in deci-degrees (x10) or centi-degrees (x100)
                # We try dividing by 100.0 first to get a sane temperature range (25C - 80C)
                
                if temp_raw > 1000 and temp_raw < 15000: # Check for common millidegree range (1C to 150C)
                    # If it looks like millidegrees, divide by 1000
                    temp_c = round(temp_raw / 1000.0, 2)
                elif temp_raw > 100 and temp_raw < 1500:
                    # If it looks like centi-degrees or deci-degrees, divide by 100
                    temp_c = round(temp_raw / 100.0, 2)
                else:
                    # Default to raw reading if it's already a small number (e.g., 35)
                    temp_c = round(temp_raw, 2)


                if temp_c > max_temp_c and temp_c < 100.0: # Ensure we log the max sensible temperature
                    max_temp_c = temp_c
        
        if max_temp_c == 0.0:
            return "N/A"
        return max_temp_c*10

    except Exception as e:
        print(f"Error reading temperature: {e}")
        return "N/A"

def get_htp_usage():
    """PLACEHOLDER for Hexagon Tensor Processor (NPU) usage."""
    # NOTE: Direct HTP/NPU usage requires specific QNN SDK commands and logging.
    # This is a placeholder. You must integrate QNN SDK profiling logic here
    # (e.g., parsing logcat output from a running QNN process, or querying
    # a dedicated QNN daemon if one is running).
    # For now, it returns a simulated value or 'N/A'.
    
    # Placeholder: Search for a running QNN process in 'top'
    # qnn_process_output = run_adb_shell_command("top -n 1 | grep qnn")
    # if qnn_process_output:
    #     return "Running"
        
    return "N/A"

# --- Main Script ---

def main():
    print(f"--- Starting QIDK Performance Monitor ---")
    print(f"Log file: {OUTPUT_FILE}")
    print(f"Interval: {SAMPLE_INTERVAL_SECONDS} seconds")

    # Initialize CSV file
    header = ['Timestamp', 'CPU Usage (%)', 'RAM Used (MB)', 'RAM Used (%)', 'Max Temperature (C)', 'HTP Usage/Status']
    try:
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    except Exception as e:
        print(f"Fatal error writing to CSV file: {e}")
        return

    # Initial run requires skipping CPU percentage calculation
    prev_cpu_data = None 
    
    # Run once to get initial CPU baseline data
    _, prev_cpu_data = get_cpu_usage(prev_cpu_data)
    
    # Print startup message and console header
    print("\nCollecting initial CPU baseline data (no output for 1st sample)...")
    print("\n" + "="*80)
    print(f"| {'TIMESTAMP':<18} | {'CPU %':<8} | {'RAM MB':<10} | {'RAM %':<6} | {'TEMP °C':<8} | {'HTP STATUS':<10} |")
    print("="*80)
    
    time.sleep(SAMPLE_INTERVAL_SECONDS)

    try:
        while True:
            # --- Collect Data ---
            cpu_percent, current_cpu_data = get_cpu_usage(prev_cpu_data)
            ram_percent, ram_used_mb = get_ram_usage()
            max_temp = get_temperature()
            htp_status = get_htp_usage()
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # --- Log Data ---
            if cpu_percent is not None:
                data_row = [
                    timestamp,
                    cpu_percent,
                    ram_used_mb,
                    ram_percent,
                    max_temp,
                    htp_status
                ]

                with open(OUTPUT_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data_row)
                
                # Print to console for real-time feedback (Aligned to match the header)
                print(f"| {timestamp:<18} | {cpu_percent:>6}% | {ram_used_mb:>10} | {ram_percent:>5}% | {max_temp:>6} | {htp_status:<10} |")
                
                # Update CPU data for the next iteration
                prev_cpu_data = current_cpu_data

            time.sleep(SAMPLE_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n--- Monitoring stopped by user (Ctrl+C) ---")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
