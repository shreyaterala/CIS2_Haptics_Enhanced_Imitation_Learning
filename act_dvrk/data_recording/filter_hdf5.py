import argparse
import h5py
import os
import sys
import glob

def list_topics(group, prefix=''):
    """Recursively list all datasets in the HDF5 file."""
    topics = []
    for key, item in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            topics.append(path)
        elif isinstance(item, h5py.Group):
            # We also consider groups as topics if the user wants to copy a whole group
            topics.append(path + "/")
            topics.extend(list_topics(item, path))
    return topics

DEFAULT_TOPICS = [
    "observations",
    "action",
    "timestamp",
    "source_timestamps",
    "arm_effort",
    "spatial_cf",
    "measured_cf",
    "joint_force" #joint_force = arm_effort - gravity_effort
    ]

def process_file(input_path, output_path, topics, force, list_only=False):
    try:
        with h5py.File(input_path, 'r') as f_in:
            if list_only:
                print(f"Topics available in '{input_path}':")
                available_topics = list_topics(f_in)
                for t in sorted(available_topics):
                    print(f"  {t}")
                return True

            if not output_path:
                print("Error: --output is required when not using --list.")
                return False

            if os.path.exists(output_path) and not force:
                print(f"Error: Output file '{output_path}' already exists. Use -f/--force to overwrite.")
                return False

            print(f"Opening '{input_path}' for reading...")
            with h5py.File(output_path, 'w') as f_out:
                print(f"Created '{output_path}' for writing...")
                
                # Also copy file-level attributes if any
                for attr_name, attr_value in f_in.attrs.items():
                    f_out.attrs[attr_name] = attr_value

                for topic in topics:
                    # Clean up trailing slashes that user might have inputted
                    clean_topic = topic.rstrip('/')
                    if clean_topic in f_in:
                        print(f"  Copying topic: {clean_topic}")
                        f_in.copy(clean_topic, f_out, clean_topic)
                    else:
                        print(f"  Warning: Topic '{clean_topic}' not found in input file.")
                
                print(f"Filtering complete. Data saved to '{output_path}'.\n")
        return True
    except OSError as e:
        print(f"Failed to process HDF5 file '{input_path}': {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Filter an HDF5 file or directory of HDF5 files to keep only specific topics.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input HDF5 file or directory")
    parser.add_argument("-o", "--output", help="Path to the output HDF5 file or directory")
    parser.add_argument("-t", "--topics", nargs='+', default=DEFAULT_TOPICS, 
                        help="List of topics (paths) to keep. Defaults to the standard dVRK topics.")
    parser.add_argument("-l", "--list", action="store_true", help="List all available topics in the input file(s) and exit")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite the output file(s) if already existing")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input '{args.input}' does not exist.")
        sys.exit(1)

    if os.path.isdir(args.input):
        if not args.list and not args.output:
            print("Error: --output directory is required when processing an input directory.")
            sys.exit(1)
            
        if args.output and not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)
            
        files = glob.glob(os.path.join(args.input, "*.hdf5")) + glob.glob(os.path.join(args.input, "*.h5"))
        if not files:
            print(f"No .hdf5 or .h5 files found in '{args.input}'.")
            sys.exit(0)
            
        print(f"Found {len(files)} file(s) to process in '{args.input}'.")
        for f in sorted(files):
            out_file = None
            if args.output:
                out_file = os.path.join(args.output, os.path.basename(f))
            success = process_file(f, out_file, args.topics, args.force, args.list)
            if not success and not args.list:
                print(f"Aborting batch processing due to error on '{f}'.")
                sys.exit(1)
    else:
        # Single file processing
        success = process_file(args.input, args.output, args.topics, args.force, args.list)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()
