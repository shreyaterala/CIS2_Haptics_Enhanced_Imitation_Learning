import argparse
import h5py
import os

def print_attrs(obj, indent=""):
    if obj.attrs:
        for key, val in obj.attrs.items():
            print(f"{indent}  @ {key}: {val}")

def print_hierarchy(obj, path='/', level=0):
    indent = '  ' * level
    
    if isinstance(obj, h5py.File):
        print(f"File: {path}")
        print_attrs(obj, indent)
        for key, val in obj.items():
            print_hierarchy(val, f"/{key}", level + 1)
            
    elif isinstance(obj, h5py.Group):
        print(f"{indent}Group: {path}")
        print_attrs(obj, indent)
        for key, val in obj.items():
            print_hierarchy(val, f"{path}/{key}" if path != '/' else f"/{key}", level + 1)
            
    elif isinstance(obj, h5py.Dataset):
        shape_str = str(obj.shape)
        dtype_str = str(obj.dtype)
        print(f"{indent}Dataset: {path} | Shape: {shape_str} | Type: {dtype_str}")
        print_attrs(obj, indent)

def print_hdf5_contents(input_path):
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' does not exist.")
        return

    try:
        with h5py.File(input_path, 'r') as f:
            print("=" * 60)
            print_hierarchy(f, input_path)
            print("=" * 60)
    except OSError as e:
        print(f"Failed to open HDF5 file '{input_path}': {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Print the contents, structure, and sizes of an HDF5 file.")
    parser.add_argument("input", help="Path to the input HDF5 file")
    
    args = parser.parse_args()
    print_hdf5_contents(args.input)

if __name__ == "__main__":
    main()
