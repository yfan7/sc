import psutil

def get_memory_usage():
    # Get system memory usage
    memory_info = psutil.virtual_memory()
    print("System memory usage:")
    print(f"Total: {memory_info.total / (1024 ** 3):.2f} GB")
    print(f"Available: {memory_info.available / (1024 ** 3):.2f} GB")
    print(f"Used: {memory_info.used / (1024 ** 3):.2f} GB")

    # Get current process memory usage
    process = psutil.Process()
    process_memory_info = process.memory_info()
    print("\nCurrent process memory usage:")
    print(f"RSS (Resident Set Size): {process_memory_info.rss / (1024 ** 2):.2f} MB")
    print(f"VMS (Virtual Memory Size): {process_memory_info.vms / (1024 ** 2):.2f} MB")

if __name__ == "__main__":
    get_memory_usage()
