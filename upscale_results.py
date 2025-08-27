import os
import argparse
import math

def upscale_list(strings, factor):
    """
    Upscales a list of strings by repeating each string 'factor' times.
    """
    if factor < 1:
        raise ValueError("Factor must be >= 1")
    
    n = len(strings)
    target_length = int(round(n * factor))
    upscaled = []

    for i in range(target_length):
        orig_index = int(math.floor(i / factor)) # Map target index back to original index
        upscaled.append(strings[orig_index])

    return upscaled


def process_files(input_folder, output_folder, factor):
    """
    Reads all .txt files in input_folder, upscales their lines, and writes
    them to output_folder with the same filenames.
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read lines
            with open(input_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            # Upscale
            upscaled_lines = upscale_list(lines, factor)

            # Write new file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(upscaled_lines))
            
            print(f"Processed: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Upscale strings in text files.")
    parser.add_argument("--input_folder", "-i", help="Folder with input text files")
    parser.add_argument("--output_folder", "-o", help="Folder to save processed text files")
    parser.add_argument("--factor", "-f", type=int, help="Upscaling factor (>=1)", default=3)
    args = parser.parse_args()

    process_files(args.input_folder, args.output_folder, args.factor)


if __name__ == "__main__":
    main()
