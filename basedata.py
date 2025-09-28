import subprocess

def getinfo():
    # Corrected command string with a space
    prompt = "adb shell dumpsys cpuinfo"

    # The shell=True argument is necessary for the command string to be interpreted correctly.
    result = subprocess.run(prompt, shell=True, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        print(f"Stderr: {result.stderr}")
        return None
    
    output = result.stdout.strip()
    return output

print(getinfo())