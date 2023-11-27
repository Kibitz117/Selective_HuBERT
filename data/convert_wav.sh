#!/bin/bash

# Base directory containing your 48kHz WAV files in fold00, fold01, and fold02
INPUT_BASE_DIR="vocal_imitation-full/48000"

# Base directory where you want to save the 16kHz WAV files in corresponding folders
OUTPUT_BASE_DIR="vocal_imitation-full/16000"

# Create the output base directory if it doesn't exist
mkdir -p "$OUTPUT_BASE_DIR"

# Process each folder
for folder in fold00 fold01 fold02; do
    # Create a directory for the output files if it doesn't exist
    mkdir -p "$OUTPUT_BASE_DIR/$folder"

    # Directory containing 48kHz files for the current folder
    INPUT_DIR="$INPUT_BASE_DIR/$folder"

    # Directory for 16kHz output files for the current folder
    OUTPUT_DIR="$OUTPUT_BASE_DIR/$folder"

    # Loop through each WAV file in the input directory
    for file in "$INPUT_DIR"/*.wav; do
        # Construct the output file path
        outfile="$OUTPUT_DIR/$(basename "$file")"

        # Convert the file to 16kHz
        ffmpeg -i "$file" -ar 16000 "$outfile"
    done
done

